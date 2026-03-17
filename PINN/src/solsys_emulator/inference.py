"""Inference APIs for trained ephemeris emulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import warnings

import astropy.units as u
import numpy as np
import torch

try:  # pragma: no cover - optional fallback for older torch.
    from torch.func import jacrev, vmap

    _HAS_TORCH_FUNC = True
except Exception:  # pragma: no cover
    jacrev = None
    vmap = None
    _HAS_TORCH_FUNC = False

from .config import FRAME_INTERNAL, FRAME_ORIGIN, TIME_SCALE
from .model import EmulatorModel
from .preprocessing import StateScaler
from .time_frames import TimeInput, build_time_grid, from_model_time, parse_time, to_model_time
from .train import load_checkpoint


class EphemerisEmulator:
    """Runtime wrapper around model + scaler + time normalization."""

    def __init__(
        self,
        model: EmulatorModel,
        scaler: StateScaler,
        bodies: Sequence[str],
        time_mean: float,
        time_std: float,
        device: str = "cpu",
        frame: str = FRAME_INTERNAL,
        origin: str = FRAME_ORIGIN,
        time_scale: str = TIME_SCALE,
        train_time_min_seconds: float | None = None,
        train_time_max_seconds: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.bodies = [str(b) for b in bodies]
        self.time_mean = float(time_mean)
        self.time_std = float(time_std) if abs(time_std) > 1e-12 else 1.0
        self.device = torch.device(device)
        self.frame = frame
        self.origin = origin
        self.time_scale = time_scale
        self.train_time_min_seconds = train_time_min_seconds
        self.train_time_max_seconds = train_time_max_seconds
        self._scaler_mean_t = torch.as_tensor(self.scaler.mean, dtype=torch.float32, device=self.device)
        self._scaler_std_t = torch.as_tensor(self.scaler.std, dtype=torch.float32, device=self.device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "cpu") -> "EphemerisEmulator":
        """Load emulator from checkpoint produced by train.py."""
        payload = load_checkpoint(checkpoint_path, map_location=device)
        model = EmulatorModel(**payload["model_kwargs"])
        model.load_state_dict(payload["model_state_dict"])
        scaler = StateScaler.from_dict(payload["scaler"])
        time_norm = payload["time_normalization"]
        bodies = payload["bodies"]
        metadata = payload.get("metadata", {})
        time_range = metadata.get("train_time_range_seconds", {})
        time_min = time_range.get("min")
        time_max = time_range.get("max")

        return cls(
            model=model,
            scaler=scaler,
            bodies=bodies,
            time_mean=float(time_norm["mean"]),
            time_std=float(time_norm["std"]),
            device=device,
            frame=str(metadata.get("frame", FRAME_INTERNAL)),
            origin=str(metadata.get("origin", FRAME_ORIGIN)),
            time_scale=str(metadata.get("timescale", TIME_SCALE)),
            train_time_min_seconds=float(time_min) if time_min is not None else None,
            train_time_max_seconds=float(time_max) if time_max is not None else None,
        )

    load = from_checkpoint

    def _predict_position_only_chunk(self, t_norm_tensor: torch.Tensor) -> np.ndarray:
        """Predict physical states [N,B,6] from a position-only model on one chunk."""
        if self.model.state_mode != "position_only":
            raise ValueError("_predict_position_only_chunk requires model.state_mode='position_only'")
        t_norm_req = t_norm_tensor.detach().clone().requires_grad_(True)
        pos_norm = self.model(t_norm_req)  # [N, B, 3]
        pos_phys = pos_norm * self._scaler_std_t[:, :, :3] + self._scaler_mean_t[:, :, :3]

        inv_time_std = 1.0 / float(self.time_std)
        if _HAS_TORCH_FUNC:
            def _single_time_pos_norm(t_scalar: torch.Tensor) -> torch.Tensor:
                return self.model(t_scalar.reshape(1))[0]

            vel_norm_time = vmap(jacrev(_single_time_pos_norm))(t_norm_req)
            vel_phys = vel_norm_time * self._scaler_std_t[:, :, :3] * inv_time_std
        else:  # pragma: no cover - legacy torch fallback.
            n_samples = int(pos_phys.shape[0])
            n_bodies = int(pos_phys.shape[1])
            vel_phys = torch.zeros((n_samples, n_bodies, 3), device=self.device, dtype=pos_phys.dtype)
            for body_idx in range(n_bodies):
                for comp_idx in range(3):
                    comp = pos_phys[:, body_idx, comp_idx]
                    dcomp_dt_norm = torch.autograd.grad(
                        comp.sum(),
                        t_norm_req,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                    vel_phys[:, body_idx, comp_idx] = dcomp_dt_norm * inv_time_std

        states = torch.cat((pos_phys, vel_phys), dim=-1)
        return states.detach().cpu().numpy()

    def _warn_if_extrapolating(self, t_seconds: np.ndarray) -> None:
        if self.train_time_min_seconds is None or self.train_time_max_seconds is None:
            return
        arr = np.atleast_1d(np.asarray(t_seconds, dtype=float))
        if np.any(arr < self.train_time_min_seconds) or np.any(arr > self.train_time_max_seconds):
            t_min_iso = from_model_time(self.train_time_min_seconds).isot
            t_max_iso = from_model_time(self.train_time_max_seconds).isot
            warnings.warn(
                (
                    "Inference requested outside training interval "
                    f"[{t_min_iso}, {t_max_iso}] ({self.time_scale.upper()}); "
                    "extrapolation errors may be very large."
                ),
                RuntimeWarning,
                stacklevel=3,
            )

    def _predict_raw(self, time_input: TimeInput | Sequence[TimeInput]) -> np.ndarray:
        t_seconds = np.asarray(to_model_time(time_input), dtype=float)
        self._warn_if_extrapolating(t_seconds)
        t_flat = np.atleast_1d(t_seconds)
        t_norm = ((t_flat - self.time_mean) / self.time_std).astype(np.float32)

        if self.model.state_mode == "position_only":
            chunk_size = 1024
            chunks: list[np.ndarray] = []
            for start in range(0, len(t_norm), chunk_size):
                stop = min(start + chunk_size, len(t_norm))
                t_chunk = torch.from_numpy(t_norm[start:stop]).to(self.device)
                chunks.append(self._predict_position_only_chunk(t_chunk))
            pred = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(self.bodies), 6), dtype=float)
        else:
            with torch.no_grad():
                t_tensor = torch.from_numpy(t_norm).to(self.device)
                pred_norm = self.model(t_tensor).cpu().numpy()
            pred = self.scaler.inverse_transform(pred_norm)
        return pred if t_seconds.ndim > 0 else pred[0]

    def predict_state(self, time_input: TimeInput) -> dict[str, dict[str, Any]]:
        """
        Predict state at one input time.

        Returns dict body -> {"r": Quantity(3), "v": Quantity(3), "frame": ..., "units": ...}
        """
        _ = parse_time(time_input)  # validates parsing + declared timescale convention
        states = self._predict_raw(time_input)
        if states.ndim != 2 or states.shape[-1] != 6:
            raise RuntimeError("predict_state expects scalar time input")

        out: dict[str, dict[str, Any]] = {}
        for idx, body in enumerate(self.bodies):
            out[body] = {
                "r": states[idx, :3] * u.km,
                "v": states[idx, 3:] * (u.km / u.s),
                "frame": self.frame,
                "origin": self.origin,
                "timescale": self.time_scale,
                "units": {"r": "km", "v": "km/s"},
            }
        return out

    def predict_trajectory(
        self,
        body: str,
        t0: TimeInput,
        t1: TimeInput,
        dt: float | u.Quantity,
    ) -> np.ndarray:
        """Predict positions [N,3] for one body between t0 and t1."""
        if isinstance(dt, u.Quantity):
            step_seconds = float(dt.to_value(u.s))
        else:
            step_seconds = float(dt)
        if step_seconds <= 0:
            raise ValueError("dt must be positive")

        body_key = body.lower()
        if body_key not in [b.lower() for b in self.bodies]:
            raise KeyError(f"Unknown body {body}")
        body_idx = [b.lower() for b in self.bodies].index(body_key)

        seconds_grid, _ = build_time_grid(t0, t1, step_seconds)
        time_grid = from_model_time(seconds_grid)
        states = self._predict_raw(time_grid)
        return np.asarray(states[:, body_idx, :3], dtype=float)
