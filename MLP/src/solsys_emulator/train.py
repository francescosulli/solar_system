"""Training utilities for the ephemeris emulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import wandb

from .config import DEFAULT_CHECKPOINT_PATH, SOFTENING_EPSILON_KM
from .constants import mu_array
from .model import EmulatorModel, ModelConfig
from .preprocessing import StateScaler, fit_scaler
from .utils import set_seed


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-6
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "cpu"
    early_stopping_patience: int | None = 30
    split_mode: str = "random"
    shuffle: bool = True
    grad_clip_norm: float | None = 1.0
    lr_scheduler: str = "cosine"
    min_lr: float = 1e-5
    physics_loss_weight: float = 0.0
    smoothness_loss_weight: float = 0.0
    nbody_loss_weight: float = 0.0
    nbody_warmup_epochs: int = 200
    nbody_softening_km: float = SOFTENING_EPSILON_KM
    nbody_relative_floor_km_s2: float = 1e-5
    position_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    show_progress: bool = True


def _split_indices(
    n_items: int,
    val_fraction: float,
    seed: int,
    split_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    if n_items < 3:
        indices = np.arange(n_items, dtype=int)
        return indices, indices

    n_val = max(1, int(round(n_items * val_fraction)))
    n_val = min(n_val, n_items - 1)
    if split_mode == "chronological":
        train_idx = np.arange(0, n_items - n_val, dtype=int)
        val_idx = np.arange(n_items - n_val, n_items, dtype=int)
        return train_idx, val_idx
    if split_mode != "random":
        raise ValueError("split_mode must be 'random' or 'chronological'")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _normalize_time_axis(times_seconds: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(times_seconds))
    std = float(np.std(times_seconds))
    if std < 1e-12:
        std = 1.0
    normalized = (times_seconds - mean) / std
    return normalized, mean, std


def _velocity_consistency_loss(times: Tensor, predictions: Tensor) -> Tensor:
    """
    Physics-inspired regularizer:
    finite-difference d(r)/dt should be consistent with predicted v.
    Inputs must be in physical units (seconds, km, km/s).
    """
    if predictions.shape[0] < 2:
        return torch.zeros((), device=predictions.device)

    order = torch.argsort(times)
    t_sorted = times[order]
    r_sorted = predictions[order, :, :3]
    v_sorted = predictions[order, :, 3:]

    dt = t_sorted[1:] - t_sorted[:-1]
    dt = torch.where(torch.abs(dt) < 1e-8, torch.full_like(dt, 1e-8), dt)
    dr = r_sorted[1:] - r_sorted[:-1]
    v_mid = 0.5 * (v_sorted[1:] + v_sorted[:-1])
    dr_dt = dr / dt[:, None, None]
    return F.mse_loss(dr_dt, v_mid)


def _trajectory_smoothness_loss(times: Tensor, predictions: Tensor) -> Tensor:
    """Finite-difference smoothness penalty on position acceleration (physical units)."""
    if predictions.shape[0] < 3:
        return torch.zeros((), device=predictions.device)

    order = torch.argsort(times)
    t_sorted = times[order]
    r_sorted = predictions[order, :, :3]

    dt1 = t_sorted[1:-1] - t_sorted[:-2]
    dt2 = t_sorted[2:] - t_sorted[1:-1]
    dt1 = torch.where(torch.abs(dt1) < 1e-8, torch.full_like(dt1, 1e-8), dt1)
    dt2 = torch.where(torch.abs(dt2) < 1e-8, torch.full_like(dt2, 1e-8), dt2)

    v1 = (r_sorted[1:-1] - r_sorted[:-2]) / dt1[:, None, None]
    v2 = (r_sorted[2:] - r_sorted[1:-1]) / dt2[:, None, None]
    dt_mid = 0.5 * (dt1 + dt2)
    accel = (v2 - v1) / dt_mid[:, None, None]
    return torch.mean(accel * accel)


def _nbody_acceleration_loss(
    times: Tensor,
    predictions: Tensor,
    mu_values: Tensor,
    softening_km: float,
    relative_floor_km_s2: float,
) -> Tensor:
    """
    Coupled N-body loss:
    finite-difference dv/dt from predictions should match gravitational acceleration
    induced by all other predicted bodies.
    """
    if predictions.shape[0] < 3:
        return torch.zeros((), device=predictions.device)

    order = torch.argsort(times)
    t_sorted = times[order]
    r_sorted = predictions[order, :, :3]
    v_sorted = predictions[order, :, 3:]

    dt_central = t_sorted[2:] - t_sorted[:-2]
    dt_central = torch.where(torch.abs(dt_central) < 1e-8, torch.full_like(dt_central, 1e-8), dt_central)
    dv_dt = (v_sorted[2:] - v_sorted[:-2]) / dt_central[:, None, None]

    r_mid = r_sorted[1:-1]  # [K, B, 3]
    r_j_minus_i = r_mid[:, None, :, :] - r_mid[:, :, None, :]  # [K, B(i), B(j), 3]
    dist_sq = torch.sum(r_j_minus_i * r_j_minus_i, dim=-1) + float(softening_km) ** 2  # [K, B, B]

    n_bodies = r_mid.shape[1]
    eye = torch.eye(n_bodies, dtype=torch.bool, device=predictions.device).unsqueeze(0)
    dist_sq = torch.where(eye, torch.full_like(dist_sq, float("inf")), dist_sq)
    inv_dist3 = torch.pow(dist_sq, -1.5)
    mu_j = mu_values.view(1, 1, n_bodies, 1)
    a_grav = torch.sum(mu_j * r_j_minus_i * inv_dist3.unsqueeze(-1), dim=2)

    # Relative residual (dimensionless): ||dv_dt - a||^2 / (||a||^2 + floor^2)
    residual = dv_dt - a_grav
    res_sq = torch.sum(residual * residual, dim=-1)
    denom_sq = torch.sum(a_grav * a_grav, dim=-1) + float(relative_floor_km_s2) ** 2
    rel = res_sq / denom_sq
    rel = torch.clamp(rel, max=1e6)
    return torch.mean(rel)


def _energy_conservation_loss(
    positions: Tensor,
    velocities: Tensor,
    mu_values: Tensor,
    softening_km: float,
) -> Tensor:
    """Variance penalty on a mass-weighted pseudo-total energy along sampled times."""
    mass_like = mu_values.view(1, -1, 1)
    kinetic = 0.5 * torch.sum(mass_like * torch.sum(velocities * velocities, dim=-1, keepdim=True), dim=1).squeeze(-1)

    r_j_minus_i = positions[:, None, :, :] - positions[:, :, None, :]
    dist = torch.sqrt(torch.sum(r_j_minus_i * r_j_minus_i, dim=-1) + float(softening_km) ** 2)
    n_bodies = positions.shape[1]
    eye = torch.eye(n_bodies, dtype=torch.bool, device=positions.device).unsqueeze(0)
    inv_dist = torch.where(eye, torch.zeros_like(dist), 1.0 / dist)
    pair_weight = (mu_values.view(1, n_bodies, 1) * mu_values.view(1, 1, n_bodies))
    potential = -0.5 * torch.sum(pair_weight * inv_dist, dim=(1, 2))

    total_energy = kinetic + potential
    energy_centered = total_energy - torch.mean(total_energy)
    denom = torch.mean(total_energy * total_energy) + 1e-12
    return torch.mean(energy_centered * energy_centered) / denom


def _angular_momentum_conservation_loss(
    positions: Tensor,
    velocities: Tensor,
    mu_values: Tensor,
) -> Tensor:
    """Variance penalty on mass-weighted pseudo-total angular momentum along sampled times."""
    mass_like = mu_values.view(1, -1, 1)
    angular_momentum = torch.sum(mass_like * torch.cross(positions, velocities, dim=-1), dim=1)
    centered = angular_momentum - torch.mean(angular_momentum, dim=0, keepdim=True)
    numer = torch.mean(torch.sum(centered * centered, dim=-1))
    denom = torch.mean(torch.sum(angular_momentum * angular_momentum, dim=-1)) + 1e-12
    return numer / denom


def _state_data_loss(
    predictions: Tensor,
    targets: Tensor,
    position_weight: float,
    velocity_weight: float,
) -> Tensor:
    """Weighted MSE between predicted and target state components."""
    pos_loss = F.mse_loss(predictions[:, :, :3], targets[:, :, :3])
    vel_loss = F.mse_loss(predictions[:, :, 3:], targets[:, :, 3:])
    return position_weight * pos_loss + velocity_weight * vel_loss


def save_checkpoint(
    path: str | Path,
    model: EmulatorModel,
    scaler: StateScaler,
    bodies: list[str],
    time_mean: float,
    time_std: float,
    metadata: dict[str, Any] | None = None,
    train_config: TrainConfig | None = None,
) -> Path:
    """Persist model and preprocessing artifacts."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.model_kwargs(),
        "scaler": scaler.to_dict(),
        "bodies": bodies,
        "time_normalization": {"mean": time_mean, "std": time_std},
        "metadata": metadata or {},
        "train_config": asdict(train_config) if train_config is not None else {},
    }
    torch.save(payload, target)
    return target


def train_emulator(
    dataset: dict[str, Any],
    train_config: TrainConfig | None = None,
    model_config: ModelConfig | None = None,
    scaler: StateScaler | None = None,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    initial_checkpoint_path: str | Path | None = None,
    use_wandb: bool = True,
    wandb_project: str = "mlp-solar-system",
) -> dict[str, Any]:
    """Train emulator on a dataset and store checkpoint.

    If ``initial_checkpoint_path`` is provided, training starts from those weights.
    """
    cfg = train_config or TrainConfig()
    set_seed(cfg.seed)

    if use_wandb:
        run_name = f"{wandb_project}_{cfg.device}_{datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "train_config": asdict(cfg),
                "model_config": asdict(model_config) if model_config else {},
            }
        )

    raw_states = np.asarray(dataset["states"], dtype=float)
    bodies = [str(b) for b in np.asarray(dataset["bodies"]).tolist()]
    times_seconds = np.asarray(dataset["times_seconds"], dtype=float)
    use_derivative_losses = (
        (cfg.physics_loss_weight > 0.0)
        or (cfg.smoothness_loss_weight > 0.0)
        or (cfg.nbody_loss_weight > 0.0)
    )

    scaler_obj = scaler or fit_scaler(raw_states)
    norm_states = scaler_obj.transform(raw_states).astype(np.float32)
    norm_times, time_mean, time_std = _normalize_time_axis(times_seconds)

    train_idx, val_idx = _split_indices(
        len(norm_times),
        cfg.val_fraction,
        cfg.seed,
        split_mode=cfg.split_mode,
    )
    if use_derivative_losses:
        train_idx = np.sort(train_idx)
    x_tensor = torch.from_numpy(norm_times.astype(np.float32))
    y_tensor = torch.from_numpy(norm_states)

    effective_shuffle = cfg.shuffle
    if use_derivative_losses and cfg.shuffle:
        # Derivative-based losses assume locally ordered times in each batch.
        effective_shuffle = False

    train_loader = DataLoader(
        TensorDataset(x_tensor[train_idx], y_tensor[train_idx]),
        batch_size=min(cfg.batch_size, len(train_idx)),
        shuffle=effective_shuffle,
    )
    val_loader = DataLoader(
        TensorDataset(x_tensor[val_idx], y_tensor[val_idx]),
        batch_size=min(cfg.batch_size, len(val_idx)),
        shuffle=False,
    )

    mcfg = model_config or ModelConfig(num_bodies=len(bodies))
    if mcfg.num_bodies != len(bodies):
        mcfg = ModelConfig(
            num_bodies=len(bodies),
            hidden_dim=mcfg.hidden_dim,
            num_layers=mcfg.num_layers,
            fourier_features=mcfg.fourier_features,
            min_frequency=mcfg.min_frequency,
            max_frequency=mcfg.max_frequency,
            frequency_spacing=mcfg.frequency_spacing,
            head_layers=mcfg.head_layers,
            head_hidden_dim=mcfg.head_hidden_dim,
            dropout=mcfg.dropout,
        )

    device = torch.device(cfg.device)
    model = EmulatorModel(**mcfg.to_kwargs()).to(device)
    if initial_checkpoint_path is not None:
        init_payload = load_checkpoint(initial_checkpoint_path, map_location=str(device))
        init_bodies = [str(b) for b in init_payload.get("bodies", [])]
        if init_bodies and init_bodies != bodies:
            raise ValueError("initial checkpoint bodies do not match current dataset bodies")
        init_model_kwargs = init_payload.get("model_kwargs", {})
        if init_model_kwargs and init_model_kwargs != mcfg.to_kwargs():
            raise ValueError("initial checkpoint model config does not match current model_config")
        model.load_state_dict(init_payload["model_state_dict"])

    scaler_mean_t = torch.as_tensor(scaler_obj.mean, dtype=torch.float32, device=device)
    scaler_std_t = torch.as_tensor(scaler_obj.std, dtype=torch.float32, device=device)
    mu_t = torch.as_tensor(mu_array(bodies), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.epochs),
            eta_min=cfg.min_lr,
        )
    elif cfg.lr_scheduler == "none":
        scheduler = None
    else:
        raise ValueError("lr_scheduler must be 'cosine' or 'none'")

    history = {
        "train_loss": [],
        "val_loss": [],
        "physics_loss": [],
        "smoothness_loss": [],
        "nbody_loss": [],
        "val_pos_rmse_km": [],
        "val_vel_rmse_km_s": [],
        "nbody_weight": [],
        "lr": [],
    }
    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0

    progress_enabled = False
    if cfg.show_progress:
        try:
            # Force std progress bar to avoid notebook-widget dependency warnings.
            from tqdm.std import tqdm

            epoch_iter = tqdm(range(cfg.epochs), desc="Training", leave=True)
            progress_enabled = True
        except Exception:  # pragma: no cover - tqdm optional at runtime.
            epoch_iter = range(cfg.epochs)
    else:
        epoch_iter = range(cfg.epochs)

    for epoch_idx in epoch_iter:
        if cfg.nbody_loss_weight > 0.0:
            warmup = max(1, int(cfg.nbody_warmup_epochs))
            nbody_weight_eff = cfg.nbody_loss_weight * min(1.0, float(epoch_idx + 1) / float(warmup))
        else:
            nbody_weight_eff = 0.0

        model.train()
        running_train = 0.0
        running_phys = 0.0
        running_smooth = 0.0
        running_nbody = 0.0
        n_batches = 0
        for batch_t, batch_y in train_loader:
            batch_t = batch_t.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_t)
            data_loss = _state_data_loss(
                pred,
                batch_y,
                position_weight=cfg.position_loss_weight,
                velocity_weight=cfg.velocity_loss_weight,
            )
            if use_derivative_losses:
                pred_phys = pred * scaler_std_t + scaler_mean_t
                batch_t_phys = batch_t * float(time_std) + float(time_mean)
                phys_loss = _velocity_consistency_loss(batch_t_phys, pred_phys)
                smooth_loss = _trajectory_smoothness_loss(batch_t_phys, pred_phys)
                nbody_loss = _nbody_acceleration_loss(
                    batch_t_phys,
                    pred_phys,
                    mu_values=mu_t,
                    softening_km=cfg.nbody_softening_km,
                    relative_floor_km_s2=cfg.nbody_relative_floor_km_s2,
                )
            else:
                phys_loss = torch.zeros((), device=device)
                smooth_loss = torch.zeros((), device=device)
                nbody_loss = torch.zeros((), device=device)
            loss = (
                data_loss
                + cfg.physics_loss_weight * phys_loss
                + cfg.smoothness_loss_weight * smooth_loss
                + nbody_weight_eff * nbody_loss
            )
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            running_train += float(data_loss.detach().cpu().item())
            running_phys += float(phys_loss.detach().cpu().item())
            running_smooth += float(smooth_loss.detach().cpu().item())
            running_nbody += float(nbody_loss.detach().cpu().item())
            n_batches += 1

        train_loss = running_train / max(1, n_batches)
        physics_loss = running_phys / max(1, n_batches)
        smoothness_loss = running_smooth / max(1, n_batches)
        nbody_loss = running_nbody / max(1, n_batches)

        model.eval()
        running_val = 0.0
        running_val_pos_rmse = 0.0
        running_val_vel_rmse = 0.0
        running_val_pos_rmse_per_body = torch.zeros(len(bodies), device=device)
        running_val_vel_rmse_per_body = torch.zeros(len(bodies), device=device)
        
        # Scientific Metrics
        running_val_energy = 0.0
        running_val_angular = 0.0
        
        val_max_pos_error = 0.0
        val_min_pos_error = float('inf')
        val_max_pos_error_per_body = torch.zeros(len(bodies), device=device)
        val_min_pos_error_per_body = torch.full((len(bodies),), float('inf'), device=device)
        
        n_val_batches = 0
        with torch.no_grad():
            for val_t, val_y in val_loader:
                val_t = val_t.to(device)
                val_y = val_y.to(device)
                val_pred = model(val_t)
                running_val += float(
                    _state_data_loss(
                        val_pred,
                        val_y,
                        position_weight=cfg.position_loss_weight,
                        velocity_weight=cfg.velocity_loss_weight,
                    ).cpu().item()
                )
                val_pred_phys = val_pred * scaler_std_t + scaler_mean_t
                val_true_phys = val_y * scaler_std_t + scaler_mean_t
                pos_diff = val_pred_phys[:, :, :3] - val_true_phys[:, :, :3]
                vel_diff = val_pred_phys[:, :, 3:] - val_true_phys[:, :, 3:]
                
                # Global RMSE
                pos_rmse = torch.sqrt(torch.mean(torch.sum(pos_diff * pos_diff, dim=-1)))
                vel_rmse = torch.sqrt(torch.mean(torch.sum(vel_diff * vel_diff, dim=-1)))
                running_val_pos_rmse += float(pos_rmse.cpu().item())
                running_val_vel_rmse += float(vel_rmse.cpu().item())
                
                # Per-body RMSE
                pos_rmse_per_body = torch.sqrt(torch.mean(torch.sum(pos_diff * pos_diff, dim=-1), dim=0))
                vel_rmse_per_body = torch.sqrt(torch.mean(torch.sum(vel_diff * vel_diff, dim=-1), dim=0))
                running_val_pos_rmse_per_body += pos_rmse_per_body
                running_val_vel_rmse_per_body += vel_rmse_per_body
                
                # Physics Conservation
                energy_err = _energy_conservation_loss(
                    positions=val_pred_phys[:, :, :3],
                    velocities=val_pred_phys[:, :, 3:],
                    mu_values=mu_t,
                    softening_km=cfg.nbody_softening_km,
                )
                angular_err = _angular_momentum_conservation_loss(
                    positions=val_pred_phys[:, :, :3],
                    velocities=val_pred_phys[:, :, 3:],
                    mu_values=mu_t,
                )
                running_val_energy += float(energy_err.cpu().item())
                running_val_angular += float(angular_err.cpu().item())
                
                # Max/Min Absolute Errors
                abs_pos_diff = torch.sqrt(torch.sum(pos_diff * pos_diff, dim=-1)) # [batch, bodies]
                
                batch_max_pos_error = float(torch.max(abs_pos_diff).cpu().item())
                batch_min_pos_error = float(torch.min(abs_pos_diff).cpu().item())
                val_max_pos_error = max(val_max_pos_error, batch_max_pos_error)
                val_min_pos_error = min(val_min_pos_error, batch_min_pos_error)
                
                batch_max_per_body = torch.max(abs_pos_diff, dim=0).values
                batch_min_per_body = torch.min(abs_pos_diff, dim=0).values
                val_max_pos_error_per_body = torch.maximum(val_max_pos_error_per_body, batch_max_per_body)
                val_min_pos_error_per_body = torch.minimum(val_min_pos_error_per_body, batch_min_per_body)
                
                n_val_batches += 1
        val_loss = running_val / max(1, n_val_batches)
        val_pos_rmse = running_val_pos_rmse / max(1, n_val_batches)
        val_vel_rmse = running_val_vel_rmse / max(1, n_val_batches)
        val_pos_rmse_per_body = running_val_pos_rmse_per_body / max(1, n_val_batches)
        val_vel_rmse_per_body = running_val_vel_rmse_per_body / max(1, n_val_batches)
        val_energy_error = running_val_energy / max(1, n_val_batches)
        val_angular_error = running_val_angular / max(1, n_val_batches)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["physics_loss"].append(physics_loss)
        history["smoothness_loss"].append(smoothness_loss)
        history["nbody_loss"].append(nbody_loss)
        history["val_pos_rmse_km"].append(val_pos_rmse)
        history["val_vel_rmse_km_s"].append(val_vel_rmse)
        history["nbody_weight"].append(float(nbody_weight_eff))
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["lr"].append(current_lr)
        
        if use_wandb:
            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pos_rmse_km": val_pos_rmse,
                "val_vel_rmse_km_s": val_vel_rmse,
                "val_energy_error": val_energy_error,
                "val_angular_error": val_angular_error,
                "val_max_pos_error_km": val_max_pos_error,
                "val_min_pos_error_km": val_min_pos_error if val_min_pos_error != float('inf') else 0.0,
                "physics_loss": physics_loss,
                "smoothness_loss": smoothness_loss,
                "nbody_loss": nbody_loss,
                "lr": current_lr,
                "epoch": epoch_idx + 1
            }
            for i, body in enumerate(bodies):
                log_dict[f"rmse_pos_{body}_km"] = float(val_pos_rmse_per_body[i].cpu().item())
                log_dict[f"rmse_vel_{body}_km_s"] = float(val_vel_rmse_per_body[i].cpu().item())
                log_dict[f"max_pos_{body}_km"] = float(val_max_pos_error_per_body[i].cpu().item())
                log_dict[f"min_pos_{body}_km"] = float(val_min_pos_error_per_body[i].cpu().item()) if val_min_pos_error_per_body[i] != float('inf') else 0.0
            wandb.log(log_dict, step=epoch_idx)
        if progress_enabled:
            epoch_iter.set_postfix(
                train=f"{train_loss:.3e}",
                val=f"{val_loss:.3e}",
                pos_rmse_km=f"{val_pos_rmse:.3e}",
                phys=f"{physics_loss:.3e}",
                smooth=f"{smoothness_loss:.3e}",
                nbody=f"{nbody_loss:.3e}",
                nbody_w=f"{nbody_weight_eff:.2e}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        elif cfg.show_progress and (epoch_idx + 1 == 1 or (epoch_idx + 1) % 25 == 0):
            print(
                (
                    f"epoch {epoch_idx + 1:04d}/{cfg.epochs} "
                    f"train={train_loss:.3e} val={val_loss:.3e} "
                    f"val_pos_rmse_km={val_pos_rmse:.3e} "
                    f"phys={physics_loss:.3e} smooth={smoothness_loss:.3e} nbody={nbody_loss:.3e} nbody_w={nbody_weight_eff:.2e} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if cfg.early_stopping_patience is not None and stale_epochs >= cfg.early_stopping_patience:
            break
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = save_checkpoint(
        checkpoint_path,
        model=model,
        scaler=scaler_obj,
        bodies=bodies,
        time_mean=time_mean,
        time_std=time_std,
        metadata={
            **dict(dataset.get("metadata", {})),
            "train_time_range_seconds": {
                "min": float(np.min(times_seconds)),
                "max": float(np.max(times_seconds)),
            },
        },
        train_config=cfg,
    )

    if use_wandb:
        wandb.run.summary["best_val_loss"] = best_val
        wandb.save(str(checkpoint_path))
        wandb.finish()

    return {
        "model": model,
        "scaler": scaler_obj,
        "time_mean": time_mean,
        "time_std": time_std,
        "history": history,
        "checkpoint_path": checkpoint,
        "bodies": bodies,
    }


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load serialized training artifacts."""
    return torch.load(Path(path), map_location=map_location)
