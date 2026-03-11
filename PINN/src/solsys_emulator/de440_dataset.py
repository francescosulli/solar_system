"""Dataset generation utilities from DE440/DE441 kernels (strict mode)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import astropy.units as u
import numpy as np
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time

from .config import (
    DEFAULT_BODIES,
    DEFAULT_DE440_URL,
    DEFAULT_KERNEL_PATH,
    FRAME_INTERNAL,
    FRAME_ORIGIN,
    MODEL_EPOCH_ISO,
    TIME_SCALE,
    ensure_project_dirs,
)
from .time_frames import TimeInput, build_time_grid, parse_time


def _normalize_body_names(bodies: Sequence[str]) -> list[str]:
    return [body.lower() for body in bodies]


def _as_time_array(time_input: TimeInput | Time) -> Time:
    times = parse_time(time_input)
    if times.isscalar:
        return Time([times.jd], format="jd", scale=times.scale)
    return times


def download_kernel(
    url: str = DEFAULT_DE440_URL,
    target_path: str | Path = DEFAULT_KERNEL_PATH,
    overwrite: bool = False,
) -> Path:
    """Download a BSP kernel into local data cache."""
    ensure_project_dirs()
    target = Path(target_path)
    if target.exists() and not overwrite:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    return target


def load_kernel(path: str | Path) -> dict[str, str]:
    """Validate and return kernel handle metadata."""
    kernel_path = Path(path)
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    # Validation step: if jplephem is installed this guarantees .bsp readability.
    try:
        from jplephem.spk import SPK  # type: ignore

        handle = SPK.open(str(kernel_path))
        handle.close()
    except ImportError:
        # jplephem is optional at runtime if astropy can still use the path.
        pass
    except Exception as exc:  # pragma: no cover - rare malformed kernel path.
        raise RuntimeError(f"Unable to open kernel with jplephem: {kernel_path}") from exc

    return {"path": str(kernel_path), "name": kernel_path.name}


def _sample_with_ephemeris(times: Time, bodies: Sequence[str], ephemeris: str) -> np.ndarray:
    states = np.zeros((len(times), len(bodies), 6), dtype=float)
    with solar_system_ephemeris.set(ephemeris):
        for body_idx, body in enumerate(bodies):
            pos, vel = get_body_barycentric_posvel(body, times)
            states[:, body_idx, :3] = pos.xyz.to_value(u.km).T
            states[:, body_idx, 3:] = vel.xyz.to_value(u.km / u.s).T
    return states


def sample_states(
    times: TimeInput | Time,
    bodies: Sequence[str] = DEFAULT_BODIES,
    kernel: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Sample barycentric states [T, B, 6] in km and km/s.

    Returns (states, metadata).
    """
    if kernel is None:
        raise ValueError("Kernel handle is required. Load DE440/DE441 with load_kernel(...).")

    body_list = _normalize_body_names(bodies)
    time_array = _as_time_array(times)
    ephemeris_name = kernel["path"]
    try:
        sampled = _sample_with_ephemeris(time_array, body_list, ephemeris_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to sample states from kernel: {ephemeris_name}") from exc
    return sampled, {"source": ephemeris_name}


def build_dataset(
    start_time: TimeInput,
    end_time: TimeInput,
    step: float | u.Quantity,
    bodies: Sequence[str] = DEFAULT_BODIES,
    kernel_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create dataset dict with states and metadata."""
    if isinstance(step, u.Quantity):
        step_seconds = float(step.to_value(u.s))
    else:
        step_seconds = float(step)

    time_seconds, time_array = build_time_grid(start_time, end_time, step_seconds)

    if kernel_path is None:
        auto_kernel = find_local_kernel()
        if auto_kernel is None:
            raise FileNotFoundError(
                "No DE440/DE441 kernel found in ./data. "
                "Place de440.bsp or de441.bsp in data/ or pass kernel_path explicitly."
            )
        kernel_path = auto_kernel
    kernel = load_kernel(kernel_path)

    states, sample_meta = sample_states(
        time_array,
        bodies=bodies,
        kernel=kernel,
    )

    body_list = _normalize_body_names(bodies)
    metadata = {
        "frame": FRAME_INTERNAL,
        "origin": FRAME_ORIGIN,
        "timescale": TIME_SCALE,
        "units": {"r": "km", "v": "km/s", "mu": "km^3/s^2"},
        "epoch": MODEL_EPOCH_ISO,
        "bodies": body_list,
        "kernel_path": str(kernel_path),
        "sample_source": sample_meta["source"],
    }
    return {
        "times_seconds": time_seconds,
        "times_iso": np.asarray(time_array.isot, dtype="U32"),
        "states": states,
        "bodies": np.asarray(body_list, dtype="U16"),
        "metadata": metadata,
    }


def save_dataset(path: str | Path, dataset: dict[str, Any]) -> Path:
    """Save dataset to compressed NPZ with JSON metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        times_seconds=np.asarray(dataset["times_seconds"], dtype=float),
        times_iso=np.asarray(dataset["times_iso"]),
        states=np.asarray(dataset["states"], dtype=float),
        bodies=np.asarray(dataset["bodies"]),
        metadata_json=json.dumps(dataset["metadata"]),
    )
    return target


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load dataset from NPZ while restoring metadata."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        raw_metadata = payload["metadata_json"].item()
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(raw_metadata)
        return {
            "times_seconds": payload["times_seconds"],
            "times_iso": payload["times_iso"],
            "states": payload["states"],
            "bodies": payload["bodies"],
            "metadata": metadata,
        }


def find_local_kernel(
    candidate_paths: Sequence[str | Path] | None = None,
) -> Path | None:
    """Find a local BSP kernel candidate in ./data."""
    candidates = list(candidate_paths or [])
    candidates.extend(
        [
            DEFAULT_KERNEL_PATH,
            DEFAULT_KERNEL_PATH.with_name("de441.bsp"),
            DEFAULT_KERNEL_PATH.with_name("de440s.bsp"),
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None
