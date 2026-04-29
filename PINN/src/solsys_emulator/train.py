"""Training utilities for the ephemeris emulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.profiler as profiler
import matplotlib.pyplot as plt

try:  # pragma: no cover - available on modern torch, optional fallback.
    from torch.func import jacrev, vmap

    _HAS_TORCH_FUNC = True
except Exception:  # pragma: no cover - old torch fallback.
    jacrev = None
    vmap = None
    _HAS_TORCH_FUNC = False

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .config import DEFAULT_CHECKPOINT_PATH, SOFTENING_EPSILON_KM
from .constants import mu_array
from .model import EmulatorModel, ModelConfig
from .preprocessing import StateScaler, fit_scaler
from .utils import set_seed

# EXTRA LOGGING
import logging
from torch.utils.tensorboard import SummaryWriter
import wandb
from .logging.distributed_logger import DistributedLogger
from .logging.logging_utils import log_config_to_tensorboard 
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
##

@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    epochs: int = 200
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-6
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "cpu"
    train_loader_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool = False
    cuda_matmul_precision: str = "high"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    early_stopping_patience: int | None = 30
    split_mode: str = "random"
    shuffle: bool = True
    grad_clip_norm: float | None = 1.0
    lr_scheduler: str = "cosine"
    min_lr: float = 1e-5
    physics_loss_weight: float = 0.0
    smoothness_loss_weight: float = 0.0
    energy_loss_weight: float = 0.0
    angular_momentum_loss_weight: float = 0.0
    physics_start_epoch: int = 0
    smoothness_start_epoch: int = 0
    energy_start_epoch: int = 0
    angular_momentum_start_epoch: int = 0
    physics_warmup_epochs: int = 1
    smoothness_warmup_epochs: int = 1
    energy_warmup_epochs: int = 1
    angular_momentum_warmup_epochs: int = 1
    nbody_loss_weight: float = 0.0
    nbody_start_epoch: int = 0
    nbody_warmup_epochs: int = 200
    nbody_softening_km: float = SOFTENING_EPSILON_KM
    nbody_relative_floor_km_s2: float = 1e-5
    adaptive_nbody_balance: bool = False
    nbody_target_fraction: float = 0.0
    nbody_balance_beta: float = 0.9
    nbody_balance_max_scale: float = 1e6
    nbody_batch_size: int | None = None
    nbody_collocation_points: int | None = None
    position_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    force_chronological_for_derivatives: bool = False
    sort_train_for_derivatives: bool = False
    compute_val_velocity_rmse: bool = True
    selection_metric: str = "val_loss"
    show_progress: bool = True
    distributed: bool = False  # Enable DDP mode
    local_rank: int = 0  # Local GPU rank (set by torchrun)
    find_unused_parameters: bool = False  # For models with conditional paths



def _compatible_model_kwargs(
    loaded: dict[str, Any],
    current: dict[str, Any],
) -> tuple[bool, str | None]:
    """
    Backward-compatible check for model kwargs.

    Older checkpoints may contain only a subset of kwargs. We require all keys
    present in the checkpoint payload to match the current configuration.
    """
    for key, value in loaded.items():
        if key not in current:
            continue
        if current[key] != value:
            return False, key
    return True, None


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


def _gravitational_acceleration_from_positions(
    positions: Tensor,
    mu_values: Tensor,
    softening_km: float,
) -> Tensor:
    """Compute coupled gravitational acceleration from positions [N, B, 3]."""
    r_j_minus_i = positions[:, None, :, :] - positions[:, :, None, :]
    dist_sq = torch.sum(r_j_minus_i * r_j_minus_i, dim=-1) + float(softening_km) ** 2

    n_bodies = positions.shape[1]
    eye = torch.eye(n_bodies, dtype=torch.bool, device=positions.device).unsqueeze(0)
    dist_sq = torch.where(eye, torch.full_like(dist_sq, float("inf")), dist_sq)
    inv_dist3 = torch.pow(dist_sq, -1.5)
    mu_j = mu_values.view(1, 1, n_bodies, 1)
    return torch.sum(mu_j * r_j_minus_i * inv_dist3.unsqueeze(-1), dim=2)


def _nbody_acceleration_residual_loss(
    positions: Tensor,
    accelerations: Tensor,
    mu_values: Tensor,
    softening_km: float,
    relative_floor_km_s2: float,
) -> Tensor:
    """
    Coupled N-body loss from autograd acceleration:
    ||a_pred - a_grav(r_pred)||^2 / (||a_grav||^2 + floor^2).
    """
    a_grav = _gravitational_acceleration_from_positions(positions, mu_values, softening_km=softening_km)
    residual = accelerations - a_grav
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


def _position_only_to_full_state_norm(
    model: EmulatorModel,
    t_norm: Tensor,
    scaler_mean_t: Tensor,
    scaler_std_t: Tensor,
    time_std_seconds: float,
    need_velocity: bool,
    need_acceleration: bool,
    create_graph: bool,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """
    Evaluate a position-only model and derive velocity/acceleration by autograd.

    Returns:
    - pred_norm_full: [N,B,6] normalized state
    - pos_phys: [N,B,3] km
    - vel_phys: [N,B,3] km/s or None
    - acc_phys: [N,B,3] km/s^2 or None
    """
    if model.state_mode != "position_only":
        raise ValueError("_position_only_to_full_state_norm requires model.state_mode='position_only'")

    if t_norm.ndim != 1:
        raise ValueError("t_norm must have shape [N]")
    if not t_norm.requires_grad:
        t_norm = t_norm.detach().clone().requires_grad_(True)

    pos_norm = model(t_norm)
    mean_pos = scaler_mean_t[:, :, :3]
    std_pos = scaler_std_t[:, :, :3]
    mean_vel = scaler_mean_t[:, :, 3:]
    std_vel = scaler_std_t[:, :, 3:]

    pos_phys = pos_norm * std_pos + mean_pos
    inv_time_std = 1.0 / float(time_std_seconds)
    inv_time_std_sq = inv_time_std * inv_time_std

    vel_phys = None
    acc_phys = None

    if need_velocity or need_acceleration:
        if _HAS_TORCH_FUNC:
            def _single_time_pos_norm(t_scalar: Tensor) -> Tensor:
                return model(t_scalar.reshape(1))[0]

            vel_norm_time = vmap(jacrev(_single_time_pos_norm))(t_norm)
            vel_phys = vel_norm_time * std_pos * inv_time_std
            if need_acceleration:
                acc_norm_time = vmap(jacrev(jacrev(_single_time_pos_norm)))(t_norm)
                acc_phys = acc_norm_time * std_pos * inv_time_std_sq
        else:  # pragma: no cover - legacy torch fallback.
            n_bodies = int(pos_phys.shape[1])
            device = pos_phys.device
            dtype = pos_phys.dtype
            vel_phys = torch.zeros((pos_phys.shape[0], n_bodies, 3), device=device, dtype=dtype)
            if need_acceleration:
                acc_phys = torch.zeros((pos_phys.shape[0], n_bodies, 3), device=device, dtype=dtype)
            for body_idx in range(n_bodies):
                for comp_idx in range(3):
                    comp = pos_phys[:, body_idx, comp_idx]
                    dcomp_dt_norm = torch.autograd.grad(
                        comp.sum(),
                        t_norm,
                        create_graph=create_graph,
                        retain_graph=True,
                    )[0]
                    vel_comp = dcomp_dt_norm * inv_time_std
                    vel_phys[:, body_idx, comp_idx] = vel_comp
                    if need_acceleration and acc_phys is not None:
                        dvel_dt_norm = torch.autograd.grad(
                            vel_comp.sum(),
                            t_norm,
                            create_graph=create_graph,
                            retain_graph=True,
                        )[0]
                        acc_phys[:, body_idx, comp_idx] = dvel_dt_norm * inv_time_std

        vel_norm = (vel_phys - mean_vel) / std_vel
    else:
        vel_norm = torch.zeros_like(mean_vel).expand(pos_norm.shape[0], -1, -1)
    pred_norm_full = torch.cat((pos_norm, vel_norm), dim=-1)
    return pred_norm_full, pos_phys, vel_phys, acc_phys


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
    enable_tensorboard: bool = True,
    use_wandb: bool = True,
    wandb_project: str = "pinn-solar-system",
    stage="unified",
    enable_profiling: bool = False,
    log_dir="logs/runs"
) -> dict[str, Any]:
    """
    Wrapper that trains a model on a given datasets, following given pecific configurations 
    for the model and the training itself,
    logs metrics and validates it  and stores checkpoint.

    If ``initial_checkpoint_path`` is provided and an existing checkpoint is found, training starts from those weights.
    """
    cfg = train_config or TrainConfig()
    set_seed(cfg.seed)

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

    ## DDP INIT ##
    rank = 0
    world_size = 1
    is_distributed = cfg.distributed
    tags = [] 
    tags.append(stage)

    if is_distributed:
        tags.append('ddp')
        if not dist.is_initialized():
            raise RuntimeError(
                "DDP mode enabled but torch.distributed not initialized. "
                "Launch with: torchrun --nproc_per_node=N train_ddp.py"
            )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = cfg.local_rank
        tags.append(f'{world_size}{device}')
        # Set device to local rank
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
        # Helper for rank-0 only printing
        def print_rank0(*args, **kwargs):
            if rank == 0:
                print(*args, **kwargs)
    else:
        device = torch.device(cfg.device)
        tags.append(f"{device}")
        print_rank0 = print
    
    print_rank0(f"[Rank {rank}/{world_size}] Training on {device}")

    ## DDP INIT ##

    mcfg = model_config or ModelConfig(num_bodies=len(bodies))
    if mcfg.num_bodies != len(bodies):
        mcfg = ModelConfig(
            num_bodies=len(bodies),
            state_mode=mcfg.state_mode,
            backbone_type=mcfg.backbone_type,
            hidden_dim=mcfg.hidden_dim,
            num_layers=mcfg.num_layers,
            fourier_features=mcfg.fourier_features,
            min_frequency=mcfg.min_frequency,
            max_frequency=mcfg.max_frequency,
            frequency_spacing=mcfg.frequency_spacing,
            head_layers=mcfg.head_layers,
            head_hidden_dim=mcfg.head_hidden_dim,
            body_embedding_dim=mcfg.body_embedding_dim,
            interaction_layers=mcfg.interaction_layers,
            interaction_hidden_dim=mcfg.interaction_hidden_dim,
            use_layer_norm=mcfg.use_layer_norm,
            dropout=mcfg.dropout,
        )
    ordered_batches_required = use_derivative_losses and mcfg.state_mode != "position_only"

    split_mode_eff = cfg.split_mode
    if ordered_batches_required and cfg.force_chronological_for_derivatives and cfg.split_mode == "random":
        split_mode_eff = "chronological"
        if cfg.show_progress:
            print("Info: derivative losses enabled, forcing split_mode='chronological'.")

    train_idx, val_idx = _split_indices(
        len(norm_times),
        cfg.val_fraction,
        cfg.seed,
        split_mode=split_mode_eff,
    )
        
    if ordered_batches_required and cfg.sort_train_for_derivatives:
        train_idx = np.sort(train_idx)
    x_tensor = torch.from_numpy(norm_times.astype(np.float32))
    y_tensor = torch.from_numpy(norm_states)

    effective_shuffle = cfg.shuffle
    if ordered_batches_required and cfg.shuffle:
        # Derivative-based losses assume locally ordered times in each batch.
        effective_shuffle = False
    if mcfg.state_mode == "position_only":
        if cfg.physics_loss_weight > 0.0:
            raise ValueError(
                "physics_loss_weight is not used with state_mode='position_only'. "
                "Use nbody_loss_weight for acceleration residual PINN training."
            )
        if cfg.smoothness_loss_weight > 0.0:
            raise ValueError("smoothness_loss_weight is not used with state_mode='position_only'.")
    if cfg.energy_loss_weight < 0.0:
        raise ValueError("energy_loss_weight must be >= 0")
    if cfg.angular_momentum_loss_weight < 0.0:
        raise ValueError("angular_momentum_loss_weight must be >= 0")
    if cfg.nbody_target_fraction < 0.0:
        raise ValueError("nbody_target_fraction must be >= 0")
    if not 0.0 <= cfg.nbody_balance_beta < 1.0:
        raise ValueError("nbody_balance_beta must be in [0, 1)")
    if cfg.nbody_balance_max_scale < 1.0:
        raise ValueError("nbody_balance_max_scale must be >= 1")
    if cfg.nbody_batch_size is not None and cfg.nbody_batch_size < 1:
        raise ValueError("nbody_batch_size must be >= 1 when provided")
    if cfg.nbody_collocation_points is not None and cfg.nbody_collocation_points < 1:
        raise ValueError("nbody_collocation_points must be >= 1 when provided")
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if cfg.train_loader_workers < 0:
        raise ValueError("train_loader_workers must be >= 0")

    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(cfg.cuda_matmul_precision))
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    pin_memory_eff = bool(cfg.pin_memory) if cfg.pin_memory is not None else device.type == "cuda"

    # train_loader_kwargs: dict[str, Any] = {
    #     "batch_size": min(cfg.batch_size, len(train_idx)),
    #     "shuffle": effective_shuffle,
    #     "num_workers": int(cfg.train_loader_workers),
    #     "pin_memory": pin_memory_eff,
    # }
    # if cfg.train_loader_workers > 0:
    #     train_loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    # train_loader = DataLoader(TensorDataset(x_tensor[train_idx], y_tensor[train_idx]), **train_loader_kwargs)

    # val_loader_kwargs: dict[str, Any] = {
    #     "batch_size": min(cfg.batch_size, len(val_idx)),
    #     "shuffle": False,
    #     "num_workers": int(cfg.train_loader_workers),
    #     "pin_memory": pin_memory_eff,
    # }
    # if cfg.train_loader_workers > 0:
    #     val_loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    # val_loader = DataLoader(TensorDataset(x_tensor[val_idx], y_tensor[val_idx]), **val_loader_kwargs)
    
    ## DDP ##
    train_dataset = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(x_tensor[val_idx], y_tensor[val_idx])

    # Distributed samplers
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=effective_shuffle,
            seed=cfg.seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        # Don't shuffle in DataLoader when using DistributedSampler
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = effective_shuffle

    # Train loader
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": min(cfg.batch_size, len(train_idx) // max(1, world_size)),
        "shuffle": train_shuffle,
        "sampler": train_sampler,
        "num_workers": int(cfg.train_loader_workers),
        "pin_memory": pin_memory_eff,
        "drop_last": True,  # Important for DDP stability
    }
    if cfg.train_loader_workers > 0:
        train_loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    # Val loader
    val_loader_kwargs: dict[str, Any] = {
        "batch_size": min(cfg.batch_size, len(val_idx) // max(1, world_size)),
        "shuffle": False,
        "sampler": val_sampler,
        "num_workers": int(cfg.train_loader_workers),
        "pin_memory": pin_memory_eff,
        "drop_last": False,
    }
    if cfg.train_loader_workers > 0:
        val_loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    ## DDP ##

    train_time_min_norm = float(np.min(norm_times[train_idx]))
    train_time_max_norm = float(np.max(norm_times[train_idx]))
    model = EmulatorModel(**mcfg.to_kwargs()).to(device)
    if initial_checkpoint_path is not None:
        init_payload = load_checkpoint(initial_checkpoint_path, map_location=str(device))
        init_bodies = [str(b) for b in init_payload.get("bodies", [])]
        if init_bodies and init_bodies != bodies:
            raise ValueError("initial checkpoint bodies do not match current dataset bodies")
        init_model_kwargs = init_payload.get("model_kwargs", {})
        if init_model_kwargs:
            ok, bad_key = _compatible_model_kwargs(init_model_kwargs, mcfg.to_kwargs())
            if not ok:
                raise ValueError(
                    "initial checkpoint model config does not match current model_config "
                    f"(mismatch on key '{bad_key}')"
                )
        model.load_state_dict(init_payload["model_state_dict"])
    # === ADD DDP WRAPPER ===
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.find_unused_parameters,
        )
        print_rank0(f"[Rank {rank}] Model wrapped with DistributedDataParallel")
    # === DDP WRAPPER ===

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
        "train_objective_loss": [],
        "val_loss": [],
        "physics_loss": [],
        "smoothness_loss": [],
        "nbody_loss": [],
        "energy_loss": [],
        "angular_momentum_loss": [],
        "physics_weight": [],
        "smoothness_weight": [],
        "energy_weight": [],
        "angular_momentum_weight": [],
        "val_pos_rmse_km": [],
        "val_vel_rmse_km_s": [],
        "nbody_weight": [],
        "lr": [],
    }
    if cfg.selection_metric not in {"val_loss", "val_pos_rmse_km"}:
        raise ValueError("selection_metric must be 'val_loss' or 'val_pos_rmse_km'")
    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    ema_train_loss: float | None = None
    ema_nbody_loss: float | None = None

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
    
    run_name = f"{wandb_project}_{stage}_{device}{world_size}_{datetime.now().strftime('%m%d_%H%M')}"

    # TENSORBOARD
    writer = None
    if rank == 0 and enable_tensorboard:
        writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

        log_config_to_tensorboard(
            writer,
            cfg=cfg,
            model_cfg=mcfg,
            world_size=world_size,
            rank=rank,
            device=str(device),
        )

        # Log hyperparameters
        # hparams = {
        #     'batch_size': cfg.batch_size,
        #     'lr': cfg.lr,
        #     'train_config': cfg,  
        #     'model_config': mcfg,
        #     'hidden_dim': mcfg.hidden_dim,
        #     'num_layers': mcfg.num_layers,
        #     'world_size': world_size,
        # }
        # writer.add_hparams(hparams, {})
        
        # Log model graph --> probably not doable/necessary ATM.
        # dummy_input = torch.randn(1, ..., device=device)
        # writer.add_graph(model, dummy_input)

    # TENSORBOARD

    # Initialize W&B (only rank 0)
    if rank == 0 and use_wandb:
        config = {
            # Model config
            'model_type': mcfg.backbone_type,
            'hidden_dim': mcfg.hidden_dim,
            'num_layers': mcfg.num_layers,
            'fourier_features': mcfg.fourier_features,
            
            # Training config
            'batch_size': cfg.batch_size,
            'effective_batch_size': cfg.batch_size * world_size,
            'lr': cfg.lr,
            'weight_decay': cfg.weight_decay,
            'epochs': cfg.epochs,
            
            # DDP config
            'world_size': world_size,
            'distributed': is_distributed,
            
            # Dataset
            'bodies': bodies,
            'num_bodies': len(bodies),
            'num_samples': len(dataset['times_seconds']),
        }
        run = wandb.init(
            project=wandb_project,
            config=config,
            name=run_name,
            tags=tags,
        )
        
        # Watch model (logs gradients and parameters)
        # wandb.watch(model, log='all', log_freq=100)
    # 

    logger = DistributedLogger(
        name="pinn_training",
        log_dir=Path("logs"),
        rank=rank,
        world_size=world_size,
        log_level=logging.DEBUG if rank == 0 else logging.INFO,
    )
    
    logger.info("Starting training...")
    logger.info(f"World size: {world_size}")
    logger.info(f"Batch size per GPU: {cfg.batch_size}")
    logger.info(f"Effective batch size: {cfg.batch_size * world_size}")

    # with profiler_ctx if profiler_ctx else nullcontext()
    for epoch_idx in epoch_iter:

        logger.info(f"Epoch {epoch_idx + 1}/{cfg.epochs}")

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        if cfg.physics_loss_weight > 0.0:
            physics_warmup = max(1, int(cfg.physics_warmup_epochs))
            start = max(0, int(cfg.physics_start_epoch))
            progress = max(0.0, float(epoch_idx + 1 - start)) / float(physics_warmup)
            physics_weight_eff = cfg.physics_loss_weight * min(1.0, progress)
        else:
            physics_weight_eff = 0.0

        if cfg.smoothness_loss_weight > 0.0:
            smooth_warmup = max(1, int(cfg.smoothness_warmup_epochs))
            start = max(0, int(cfg.smoothness_start_epoch))
            progress = max(0.0, float(epoch_idx + 1 - start)) / float(smooth_warmup)
            smooth_weight_eff = cfg.smoothness_loss_weight * min(1.0, progress)
        else:
            smooth_weight_eff = 0.0

        if cfg.energy_loss_weight > 0.0:
            energy_warmup = max(1, int(cfg.energy_warmup_epochs))
            start = max(0, int(cfg.energy_start_epoch))
            progress = max(0.0, float(epoch_idx + 1 - start)) / float(energy_warmup)
            energy_weight_eff = cfg.energy_loss_weight * min(1.0, progress)
        else:
            energy_weight_eff = 0.0

        if cfg.angular_momentum_loss_weight > 0.0:
            ang_warmup = max(1, int(cfg.angular_momentum_warmup_epochs))
            start = max(0, int(cfg.angular_momentum_start_epoch))
            progress = max(0.0, float(epoch_idx + 1 - start)) / float(ang_warmup)
            angular_weight_eff = cfg.angular_momentum_loss_weight * min(1.0, progress)
        else:
            angular_weight_eff = 0.0

        if cfg.nbody_loss_weight > 0.0:
            warmup = max(1, int(cfg.nbody_warmup_epochs))
            start = max(0, int(cfg.nbody_start_epoch))
            progress = max(0.0, float(epoch_idx + 1 - start)) / float(warmup)
            nbody_weight_eff = cfg.nbody_loss_weight * min(1.0, progress)
            if (
                cfg.adaptive_nbody_balance
                and cfg.nbody_target_fraction > 0.0
                and ema_train_loss is not None
                and ema_nbody_loss is not None
                and ema_nbody_loss > 1e-20
                and nbody_weight_eff > 0.0
            ):
                target_weight = cfg.nbody_target_fraction * max(ema_train_loss, 1e-20) / max(ema_nbody_loss, 1e-20)
                max_weight = nbody_weight_eff * float(cfg.nbody_balance_max_scale)
                nbody_weight_eff = min(max_weight, max(nbody_weight_eff, target_weight))
        else:
            nbody_weight_eff = 0.0

        model.train()
        running_train = 0.0
        running_objective = 0.0
        running_phys = 0.0
        running_smooth = 0.0
        running_nbody = 0.0
        running_energy = 0.0
        running_angular = 0.0
        n_batches = 0
        accum_steps = max(1, int(cfg.gradient_accumulation_steps))
        optimizer.zero_grad(set_to_none=True)

        logger.info(
            "Initiating logging helpers...\n"
            f"TENSORBOARD={enable_tensorboard} \n "
            f"WANDB={use_wandb} \n"
            f"PROFILING={enable_profiling} \n"
            )

        for batch_idx, (batch_t, batch_y) in enumerate(train_loader):

            # Log batch stats (debug)
            if batch_idx % 100 == 0:
                logger.log_batch_stats(batch_t, batch_idx)

            batch_t = batch_t.to(device, non_blocking=pin_memory_eff)
            batch_y = batch_y.to(device, non_blocking=pin_memory_eff)
            if model.state_mode == "position_only":
                need_acc = nbody_weight_eff > 0.0
                need_energy = energy_weight_eff > 0.0
                need_angular = angular_weight_eff > 0.0
                use_collocation = (need_acc or need_energy or need_angular) and cfg.nbody_collocation_points is not None
                use_nbody_subset = (
                    need_acc
                    and not use_collocation
                    and cfg.nbody_batch_size is not None
                    and int(cfg.nbody_batch_size) < int(batch_t.shape[0])
                )
                need_vel = cfg.velocity_loss_weight > 0.0 or ((need_energy or need_angular) and not use_collocation)
                need_batch_acc = need_acc and not use_collocation and not use_nbody_subset
                pred, pos_phys, vel_phys, acc_phys = _position_only_to_full_state_norm(
                    model=model,
                    t_norm=batch_t.detach().clone().requires_grad_(True),
                    scaler_mean_t=scaler_mean_t,
                    scaler_std_t=scaler_std_t,
                    time_std_seconds=float(time_std),
                    need_velocity=need_vel,
                    need_acceleration=need_batch_acc,
                    create_graph=True,
                )
                data_loss = _state_data_loss(
                    pred,
                    batch_y,
                    position_weight=cfg.position_loss_weight,
                    velocity_weight=cfg.velocity_loss_weight,
                )
                phys_loss = torch.zeros((), device=device)
                smooth_loss = torch.zeros((), device=device)
                energy_loss = torch.zeros((), device=device)
                angular_momentum_loss = torch.zeros((), device=device)
                if use_collocation:
                    collocation_size = int(cfg.nbody_collocation_points)
                    coll_t = torch.empty(collocation_size, device=device, dtype=batch_t.dtype).uniform_(
                        train_time_min_norm,
                        train_time_max_norm,
                    )
                    coll_t = coll_t.requires_grad_(True)
                    _, pos_phys_coll, vel_phys_coll, acc_phys_coll = _position_only_to_full_state_norm(
                        model=model,
                        t_norm=coll_t,
                        scaler_mean_t=scaler_mean_t,
                        scaler_std_t=scaler_std_t,
                        time_std_seconds=float(time_std),
                        need_velocity=True,
                        need_acceleration=need_acc,
                        create_graph=True,
                    )
                    if need_acc:
                        nbody_loss = _nbody_acceleration_residual_loss(
                            positions=pos_phys_coll,
                            accelerations=acc_phys_coll,
                            mu_values=mu_t,
                            softening_km=cfg.nbody_softening_km,
                            relative_floor_km_s2=cfg.nbody_relative_floor_km_s2,
                        )
                    else:
                        nbody_loss = torch.zeros((), device=device)
                    if need_energy:
                        energy_loss = _energy_conservation_loss(
                            positions=pos_phys_coll,
                            velocities=vel_phys_coll,
                            mu_values=mu_t,
                            softening_km=cfg.nbody_softening_km,
                        )
                    if need_angular:
                        angular_momentum_loss = _angular_momentum_conservation_loss(
                            positions=pos_phys_coll,
                            velocities=vel_phys_coll,
                            mu_values=mu_t,
                        )
                elif need_acc and acc_phys is not None:
                    nbody_loss = _nbody_acceleration_residual_loss(
                        positions=pos_phys,
                        accelerations=acc_phys,
                        mu_values=mu_t,
                        softening_km=cfg.nbody_softening_km,
                        relative_floor_km_s2=cfg.nbody_relative_floor_km_s2,
                    )
                elif need_acc and use_nbody_subset:
                    subset_size = min(int(cfg.nbody_batch_size), int(batch_t.shape[0]))
                    subset_idx = torch.randperm(int(batch_t.shape[0]), device=batch_t.device)[:subset_size]
                    subset_t = batch_t.index_select(0, subset_idx).detach().clone().requires_grad_(True)
                    _, pos_phys_subset, _, acc_phys_subset = _position_only_to_full_state_norm(
                        model=model,
                        t_norm=subset_t,
                        scaler_mean_t=scaler_mean_t,
                        scaler_std_t=scaler_std_t,
                        time_std_seconds=float(time_std),
                        need_velocity=True,
                        need_acceleration=True,
                        create_graph=True,
                    )
                    nbody_loss = _nbody_acceleration_residual_loss(
                        positions=pos_phys_subset,
                        accelerations=acc_phys_subset,
                        mu_values=mu_t,
                        softening_km=cfg.nbody_softening_km,
                        relative_floor_km_s2=cfg.nbody_relative_floor_km_s2,
                    )
                else:
                    nbody_loss = torch.zeros((), device=device)

                if not use_collocation:
                    if need_energy and vel_phys is not None:
                        energy_loss = _energy_conservation_loss(
                            positions=pos_phys,
                            velocities=vel_phys,
                            mu_values=mu_t,
                            softening_km=cfg.nbody_softening_km,
                        )
                    if need_angular and vel_phys is not None:
                        angular_momentum_loss = _angular_momentum_conservation_loss(
                            positions=pos_phys,
                            velocities=vel_phys,
                            mu_values=mu_t,
                        )
            else:
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
                energy_loss = torch.zeros((), device=device)
                angular_momentum_loss = torch.zeros((), device=device)
            loss = (
                data_loss
                + physics_weight_eff * phys_loss
                + smooth_weight_eff * smooth_loss
                + nbody_weight_eff * nbody_loss
                + energy_weight_eff * energy_loss
                + angular_weight_eff * angular_momentum_loss
            )
            scaled_loss = loss / float(accum_steps)
            scaled_loss.backward()
            if ((n_batches + 1) % accum_steps == 0) or ((n_batches + 1) == len(train_loader)):
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Log gradients
            if batch_idx % 100 == 0:
                logger.log_gradients(model, batch_idx)
            
            # Log GPU memory
            if batch_idx % 100 == 0:
                logger.log_gpu_memory(batch_idx)

            ## WANDB
            if rank == 0 and use_wandb:
                # Save history dictionary as artifact
                artifact = wandb.Artifact(
                    name=f'training_history_epoch_{epoch_idx}',
                    type='training_history'
                )
                
                # Save as JSON --> probably unnecessarily verbose
                # import json
                # history_path = f'history_epoch_{epoch_idx}.json'
                # with open(history_path, 'w') as f:
                #     json.dump(history, f, indent=2)
                # artifact.add_file(history_path)
                
                # wandb.log_artifact(artifact)
            ## WANDB

        running_train += float(data_loss.detach().cpu().item())
        running_objective += float(loss.detach().cpu().item())
        running_phys += float(phys_loss.detach().cpu().item())
        running_smooth += float(smooth_loss.detach().cpu().item())
        running_nbody += float(nbody_loss.detach().cpu().item())
        running_energy += float(energy_loss.detach().cpu().item())
        running_angular += float(angular_momentum_loss.detach().cpu().item())
        n_batches += 1

        # === ADD METRIC AGGREGATION ===
        if is_distributed:
            # Stack all metrics that need aggregation
            metrics_tensor = torch.tensor(
                [
                    running_train,
                    running_objective,
                    running_phys,
                    running_smooth,
                    running_nbody,
                    running_energy,
                    running_angular,
                    float(n_batches),
                ],
                dtype=torch.float32,
                device=device,
            )
            
            # Sum across all ranks
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            
            # Unpack aggregated metrics
            (
                running_train,
                running_objective,
                running_phys,
                running_smooth,
                running_nbody,
                running_energy,
                running_angular,
                n_batches_total,
            ) = metrics_tensor.tolist()
            
            n_batches = int(n_batches_total)

        train_loss = running_train / max(1, n_batches)
        train_objective_loss = running_objective / max(1, n_batches)
        physics_loss = running_phys / max(1, n_batches)
        smoothness_loss = running_smooth / max(1, n_batches)
        nbody_loss = running_nbody / max(1, n_batches)
        energy_loss = running_energy / max(1, n_batches)
        angular_momentum_loss = running_angular / max(1, n_batches)
        
        if cfg.adaptive_nbody_balance and cfg.nbody_target_fraction > 0.0:
            beta = float(cfg.nbody_balance_beta)
            if ema_train_loss is None:
                ema_train_loss = train_loss
            else:
                ema_train_loss = beta * ema_train_loss + (1.0 - beta) * train_loss
            if nbody_loss > 0.0:
                if ema_nbody_loss is None:
                    ema_nbody_loss = nbody_loss
                else:
                    ema_nbody_loss = beta * ema_nbody_loss + (1.0 - beta) * nbody_loss

        # TODO: this part concerns calidation and should probably be separated from the rest, also to reduce the size of this gigantic function and make everything a bit more modular
        model.eval()
        running_val = 0.0
        running_val_pos_rmse = 0.0
        running_val_vel_rmse = 0.0
        n_val_batches = 0
        if model.state_mode == "position_only":
            for val_t, val_y in val_loader:
                val_t = val_t.to(device, non_blocking=pin_memory_eff)
                val_y = val_y.to(device, non_blocking=pin_memory_eff)
                need_val_vel = cfg.velocity_loss_weight > 0.0 or cfg.compute_val_velocity_rmse
                val_pred, val_pos_phys, val_vel_phys, _ = _position_only_to_full_state_norm(
                    model=model,
                    t_norm=val_t.detach().clone().requires_grad_(True),
                    scaler_mean_t=scaler_mean_t,
                    scaler_std_t=scaler_std_t,
                    time_std_seconds=float(time_std),
                    need_velocity=need_val_vel,
                    need_acceleration=False,
                    create_graph=False,
                )
                running_val += float(
                    _state_data_loss(
                        val_pred,
                        val_y,
                        position_weight=cfg.position_loss_weight,
                        velocity_weight=cfg.velocity_loss_weight,
                    ).detach().cpu().item()
                )
                val_true_phys = val_y * scaler_std_t + scaler_mean_t
                pos_diff = val_pos_phys - val_true_phys[:, :, :3]
                pos_rmse = torch.sqrt(torch.mean(torch.sum(pos_diff * pos_diff, dim=-1)))
                running_val_pos_rmse += float(pos_rmse.detach().cpu().item())
                if val_vel_phys is not None:
                    vel_diff = val_vel_phys - val_true_phys[:, :, 3:]
                    vel_rmse = torch.sqrt(torch.mean(torch.sum(vel_diff * vel_diff, dim=-1)))
                    running_val_vel_rmse += float(vel_rmse.detach().cpu().item())
                else:
                    running_val_vel_rmse += float("nan")
                n_val_batches += 1
        else:
            with torch.no_grad():
                for val_t, val_y in val_loader:
                    val_t = val_t.to(device, non_blocking=pin_memory_eff)
                    val_y = val_y.to(device, non_blocking=pin_memory_eff)
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
                    pos_rmse = torch.sqrt(torch.mean(torch.sum(pos_diff * pos_diff, dim=-1)))
                    vel_rmse = torch.sqrt(torch.mean(torch.sum(vel_diff * vel_diff, dim=-1)))
                    running_val_pos_rmse += float(pos_rmse.cpu().item())
                    running_val_vel_rmse += float(vel_rmse.cpu().item())
                    n_val_batches += 1

                ## DDP ##
                if is_distributed:
                    val_metrics_tensor = torch.tensor(
                        [val_running, val_pos_error_sum, val_vel_error_sum, float(val_count)],
                        dtype=torch.float32,
                        device=device,
                    )
                    dist.all_reduce(val_metrics_tensor, op=dist.ReduceOp.SUM)
                    val_running, val_pos_error_sum, val_vel_error_sum, val_count_total = (
                        val_metrics_tensor.tolist()
                    )
                    val_count = int(val_count_total)

        epoch_val_loss = running_val/ max(1, len(val_loader) * world_size)
        epoch_val_pos_rmse = running_val_pos_rmse/ max(1, len(val_loader) * world_size)
        epoch_val_vel_rmse = running_val_vel_rmse/ max(1, len(val_loader) * world_size)

                ## DDP ##

        val_loss = running_val / max(1, n_val_batches)
        val_pos_rmse = running_val_pos_rmse / max(1, n_val_batches)
        if model.state_mode == "position_only" and not cfg.compute_val_velocity_rmse and cfg.velocity_loss_weight <= 0.0:
            val_vel_rmse = float("nan")
        else:
            val_vel_rmse = running_val_vel_rmse / max(1, n_val_batches)

        logger.info(
            f"Epoch {epoch_idx + 1} complete | "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={epoch_val_loss:.6f}"
            )


        ## TENSORBOARD
        if rank == 0 and writer is not None:
            # Scalars
            writer.add_scalar('Loss/train', train_loss, epoch_idx)
            writer.add_scalar('Loss/val', epoch_val_loss, epoch_idx)
            writer.add_scalar('RMSE/position', val_pos_rmse, epoch_idx)
            writer.add_scalar('RMSE/velocity', val_vel_rmse, epoch_idx)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch_idx)
            
            # Physics losses
            writer.add_scalar('Physics/nbody_raw', nbody_loss, epoch_idx)
            writer.add_scalar('Physics/nbody_weight', nbody_weight_eff, epoch_idx)
            
            # Gradients
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    writer.add_scalar(f'Gradients/{name}', param_norm, epoch_idx)
            
            total_norm = total_norm ** 0.5
            writer.add_scalar('Gradients/total_norm', total_norm, epoch_idx)
            
            # Memory usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    writer.add_scalar(f'Memory/gpu_{i}_allocated_GB', mem_allocated, epoch_idx)
                    writer.add_scalar(f'Memory/gpu_{i}_reserved_GB', mem_reserved, epoch_idx)
            
            # Histograms (every N epochs to reduce overhead)
            if epoch_idx % 10 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Parameters/{name}', param, epoch_idx)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch_idx)
            
            writer.flush()
        ## TENSORBOARD

        ## WANDB
        if rank == 0 and use_wandb:
            metrics = {
                'epoch': epoch_idx,
                'train/loss': train_loss,
                'train/objective_loss': train_objective_loss,
                'val/loss': epoch_val_loss,
                'val/pos_rmse_km': val_pos_rmse,
                'val/vel_rmse_km_s': val_vel_rmse,
                
                # Add the missing physics metrics
                'physics/total_loss': physics_loss,
                'physics/smoothness_loss': smoothness_loss,
                'physics/nbody_loss': nbody_loss,
                'physics/nbody_weight': nbody_weight_eff,
                'physics/energy_loss': energy_loss,
                'physics/angular_momentum_loss': angular_momentum_loss,
                
                # Add the missing weights
                'weights/physics': physics_weight_eff,
                'weights/smoothness': smooth_weight_eff,
                'weights/energy': energy_weight_eff,
                'weights/angular_momentum': angular_weight_eff,
                
                'optim/lr': optimizer.param_groups[0]['lr'],
            }
            
            # GPU metrics (all devices)
            for i in range(torch.cuda.device_count()):
                metrics[f'gpu_{i}/memory_allocated_GB'] = \
                    torch.cuda.memory_allocated(i) / 1024**3
                metrics[f'gpu_{i}/memory_reserved_GB'] = \
                    torch.cuda.memory_reserved(i) / 1024**3
            
            wandb.log(metrics, step=epoch_idx)
            
            # Log images (every N epochs)
            if epoch_idx % 50 == 0:
                # Plot training curves
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history['train_loss'], label='Train')
                ax.plot(history['val_loss'], label='Val')
                ax.set_yscale('log')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                wandb.log({"training_curves": wandb.Image(fig)}, step=epoch_idx)
                plt.close(fig)
        # WANDB

        # PYTORCH MEMORY PROFILER
        if not enable_profiling or rank != 0:
            profiler_ctx = None
        else:
            profiler_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=2
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
        # PYTORCH MEMORY PROFILER

        history["train_loss"].append(train_loss)
        history["train_objective_loss"].append(train_objective_loss)
        history["val_loss"].append(val_loss)
        history["physics_loss"].append(physics_loss)
        history["smoothness_loss"].append(smoothness_loss)
        history["nbody_loss"].append(nbody_loss)
        history["energy_loss"].append(energy_loss)
        history["angular_momentum_loss"].append(angular_momentum_loss)
        history["physics_weight"].append(float(physics_weight_eff))
        history["smoothness_weight"].append(float(smooth_weight_eff))
        history["energy_weight"].append(float(energy_weight_eff))
        history["angular_momentum_weight"].append(float(angular_weight_eff))
        history["val_pos_rmse_km"].append(val_pos_rmse)
        history["val_vel_rmse_km_s"].append(val_vel_rmse)
        history["nbody_weight"].append(float(nbody_weight_eff))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if progress_enabled:
            if model.state_mode == "position_only":
                postfix = {
                    "obj": f"{train_objective_loss:.3e}",
                    "data": f"{train_loss:.3e}",
                    "val": f"{val_loss:.3e}",
                    "pos_rmse_km": f"{val_pos_rmse:.3e}",
                    "nbody": f"{nbody_loss:.3e}",
                    "nbody_w": f"{nbody_weight_eff:.2e}",
                    "energy": f"{energy_loss:.3e}",
                    "angmom": f"{angular_momentum_loss:.3e}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                if np.isfinite(val_vel_rmse):
                    postfix["vel_rmse_km_s"] = f"{val_vel_rmse:.3e}"
            else:
                postfix = {
                    "obj": f"{train_objective_loss:.3e}",
                    "data": f"{train_loss:.3e}",
                    "val": f"{val_loss:.3e}",
                    "pos_rmse_km": f"{val_pos_rmse:.3e}",
                    "phys": f"{physics_loss:.3e}",
                    "smooth": f"{smoothness_loss:.3e}",
                    "nbody": f"{nbody_loss:.3e}",
                    "phys_w": f"{physics_weight_eff:.2e}",
                    "nbody_w": f"{nbody_weight_eff:.2e}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            epoch_iter.set_postfix(**postfix)
        elif cfg.show_progress and (epoch_idx + 1 == 1 or (epoch_idx + 1) % 25 == 0):
            if model.state_mode == "position_only":
                msg = (
                    f"epoch {epoch_idx + 1:04d}/{cfg.epochs} "
                    f"obj={train_objective_loss:.3e} data={train_loss:.3e} val={val_loss:.3e} "
                    f"val_pos_rmse_km={val_pos_rmse:.3e} nbody={nbody_loss:.3e} "
                    f"energy={energy_loss:.3e} angmom={angular_momentum_loss:.3e} "
                    f"nbody_w={nbody_weight_eff:.2e} lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                if np.isfinite(val_vel_rmse):
                    msg += f" val_vel_rmse_km_s={val_vel_rmse:.3e}"
            else:
                msg = (
                    f"epoch {epoch_idx + 1:04d}/{cfg.epochs} "
                    f"obj={train_objective_loss:.3e} data={train_loss:.3e} val={val_loss:.3e} "
                    f"val_pos_rmse_km={val_pos_rmse:.3e} phys={physics_loss:.3e} "
                    f"phys_w={physics_weight_eff:.2e} smooth={smoothness_loss:.3e} "
                    f"nbody={nbody_loss:.3e} nbody_w={nbody_weight_eff:.2e} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
            print(msg)

        metric_value = val_loss if cfg.selection_metric == "val_loss" else val_pos_rmse
        if metric_value < best_val:
            best_val = metric_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
            # === MODIFIED: Get state dict correctly for DDP ===
            if is_distributed:
                # DDP wraps the model, access .module to get underlying model
                best_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
            else:
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
            stale_epochs = 0
            print_rank0(f"  → New best {cfg.selection_metric}: {best_val:.6f}")

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
    # === MODIFIED: Save only from rank 0 ===
    if rank == 0:
        if best_state is None:
            if is_distributed:
                best_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
            else:
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        save_payload = {
            "model_state_dict": best_state,
            "model_kwargs": mcfg.to_kwargs(),
            "scaler_mean": scaler_obj.mean,
            "scaler_std": scaler_obj.std,
            "bodies": bodies,
            "history": history,
        }
        torch.save(save_payload, checkpoint_path)
        print_rank0(f"✓ Checkpoint saved: {checkpoint_path}")
    


    # === ADD BARRIER ===
    # Ensure all ranks wait for rank 0 to finish saving
    if is_distributed:
        dist.barrier()
    # TENSORBOARD
    if rank == 0 and writer is not None:
        writer.close()
    # TENSORBOARD

    # WANDB
    if rank == 0 and use_wandb:
        # Save final checkpoint to W&B
        wandb.save(str(checkpoint_path))
        
        # Log summary metrics
        wandb.run.summary['best_val_loss'] = best_val
        wandb.run.summary['total_epochs'] = epoch_idx + 1
        
        wandb.finish()
    # WANDB
    
    # DISTRIBUTED LOGGER
    logger.sync_and_log("Training complete!")
    ## 
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
