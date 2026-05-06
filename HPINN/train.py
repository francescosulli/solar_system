#!/usr/bin/env python3
"""
Unified training entrypoint for the Hybrid PINN Solar System emulator.

HPINN keeps the same dataset/API as PINN but uses a hybrid correction term:
a physics-first position-only PINN plus a small learned residual branch.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from solsys_emulator.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATASET_PATH, ensure_project_dirs
from solsys_emulator.de440_dataset import load_dataset
from solsys_emulator.model import ModelConfig
from solsys_emulator.train import TrainConfig, train_emulator


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training script for Solar System HPINN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  unified:    Single-stage training with unified configuration (default)
  multi:      Three-stage training (coarse → refine → physics) with checkpointing
  
Examples:
  # Single-stage GPU training
  python train.py --training-mode unified --device cuda --epochs 1400
  
  # Multi-stage training with resumption support
  python train.py --training-mode multi --device cuda
  
  # Multi-stage, skip physics stage
  python train.py --training-mode multi --skip-physics-stage
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--training-mode",
        type=str,
        default="unified",
        choices=("unified", "multi"),
        help="Training mode: 'unified' (single-stage) or 'multi' (three-stage with checkpointing)",
    )
    
    # Common arguments
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="Input dataset .npz path")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Final checkpoint path")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. 'auto' selects cuda when available",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="LEOPARDD",
        help="Name of the project for logs",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=None,
        help="Override DataLoader workers. Default: 8 on cuda, 0 on cpu",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for training figures and summary JSON. Defaults to checkpoint parent dir",
    )
    
    # Unified mode arguments
    unified_group = parser.add_argument_group("unified mode options")
    unified_group.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    unified_group.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    unified_group.add_argument(
        "--collocation-points",
        type=int,
        default=None,
        help="Override collocation points per batch for n-body residual",
    )
    
    # Multi-stage mode arguments
    multi_group = parser.add_argument_group("multi-stage mode options")
    multi_group.add_argument(
        "--skip-physics-stage",
        action="store_true",
        help="Skip the final n-body PINN fine-tuning stage (multi mode only)",
    )
    
    return parser.parse_args()


# ============================================================================
# Device and Memory Management
# ============================================================================
import os
import torch
import torch.distributed as dist

def resolve_device(device_arg: str) -> torch.device:
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Setting cuda to: {local_rank}")
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available")
        return torch.device("cuda:0")

    return torch.device(device_arg)

## OLD, not fit for multigpu
# def resolve_device(device_arg: str) -> str:
#     """Resolve device string to actual device."""
#     if device_arg == "auto":
#         return "cuda" if torch.cuda.is_available() else "cpu"
#     if device_arg == "cuda" and not torch.cuda.is_available():
#         raise RuntimeError("Requested device 'cuda' but CUDA is not available")
#     return device_arg


def clear_gpu_memory() -> None:
    """Aggressively clear GPU memory between training stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory_summary() -> None:
    """Print current GPU memory usage (if CUDA available)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


# ============================================================================
# Configuration Builders
# ============================================================================

def build_unified_profile(
    dataset: dict[str, Any],
    device: str,
    loader_workers_override: int | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    collocation_override: int | None = None,
) -> tuple[ModelConfig, TrainConfig, dict[str, Any]]:
    """Build configuration for unified single-stage training."""
    use_gpu_profile = device == "cuda"
    is_distributed = True if torch.cuda.device_count() > 1 else False 
    loader_workers = loader_workers_override if loader_workers_override is not None else (8 if use_gpu_profile else 0)
    pin_memory = True if use_gpu_profile else None
    persistent_workers = True if use_gpu_profile and loader_workers > 0 else False

    if use_gpu_profile:
        model_cfg = ModelConfig(
            num_bodies=len(dataset["bodies"]),
            state_mode="position_only",
            backbone_type="residual",
            hidden_dim=640,
            num_layers=8,
            fourier_features=80,
            min_frequency=0.02,
            max_frequency=96.0,
            frequency_spacing="log",
            head_layers=3,
            head_hidden_dim=320,
            body_embedding_dim=64,
            interaction_layers=2,
            interaction_hidden_dim=256,
            hybrid_correction=True,
            correction_layers=2,
            correction_hidden_dim=128,
            correction_init_scale=0.02,
            use_layer_norm=True,
            dropout=0.0,
        )
        train_cfg = TrainConfig(
            epochs=epochs_override or 1400,
            batch_size=batch_size_override or 768,
            gradient_accumulation_steps=2,
            lr=2.5e-4,
            weight_decay=5e-7,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
            train_loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            early_stopping_patience=260,
            lr_scheduler="cosine",
            min_lr=5e-7,
            nbody_loss_weight=4e-6,
            adaptive_nbody_balance=True,
            nbody_target_fraction=1e-2,
            nbody_balance_beta=0.97,
            nbody_balance_max_scale=1e8,
            nbody_collocation_points=collocation_override or 192,
            nbody_start_epoch=40,
            nbody_warmup_epochs=320,
            nbody_softening_km=50_000.0,
            nbody_relative_floor_km_s2=2e-4,
            physics_loss_weight=0.0,
            smoothness_loss_weight=0.0,
            energy_loss_weight=2e-4,
            angular_momentum_loss_weight=2e-4,
            correction_loss_weight=7.5e-4,
            energy_start_epoch=40,
            angular_momentum_start_epoch=40,
            correction_start_epoch=20,
            energy_warmup_epochs=320,
            angular_momentum_warmup_epochs=320,
            correction_warmup_epochs=200,
            position_loss_weight=1.0,
            velocity_loss_weight=0.5,
            grad_clip_norm=1.0,
            compute_val_velocity_rmse=True,
            selection_metric="val_pos_rmse_km",
            force_chronological_for_derivatives=False,
            sort_train_for_derivatives=False,
            show_progress=True,
            device=device,
            cuda_matmul_precision="high",
            allow_tf32=True,
            cudnn_benchmark=True,
            distributed=is_distributed
        )
    else:
        model_cfg = ModelConfig(
            num_bodies=len(dataset["bodies"]),
            state_mode="position_only",
            backbone_type="plain",
            hidden_dim=384,
            num_layers=6,
            fourier_features=40,
            min_frequency=0.25,
            max_frequency=48.0,
            frequency_spacing="log",
            head_layers=2,
            head_hidden_dim=160,
            body_embedding_dim=0,
            interaction_layers=0,
            interaction_hidden_dim=128,
            hybrid_correction=True,
            correction_layers=2,
            correction_hidden_dim=64,
            correction_init_scale=0.02,
            use_layer_norm=False,
            dropout=0.0,
        )
        train_cfg = TrainConfig(
            epochs=epochs_override or 600,
            batch_size=batch_size_override or 384,
            gradient_accumulation_steps=1,
            lr=2e-4,
            weight_decay=1e-6,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
            train_loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            early_stopping_patience=120,
            lr_scheduler="cosine",
            min_lr=5e-7,
            nbody_loss_weight=1.2e-6,
            adaptive_nbody_balance=True,
            nbody_target_fraction=3e-3,
            nbody_balance_beta=0.95,
            nbody_balance_max_scale=1e6,
            nbody_collocation_points=collocation_override or 64,
            nbody_start_epoch=20,
            nbody_warmup_epochs=140,
            nbody_softening_km=80_000.0,
            nbody_relative_floor_km_s2=5e-4,
            physics_loss_weight=0.0,
            smoothness_loss_weight=0.0,
            energy_loss_weight=5e-5,
            angular_momentum_loss_weight=5e-5,
            correction_loss_weight=2e-4,
            energy_start_epoch=20,
            angular_momentum_start_epoch=20,
            correction_start_epoch=10,
            energy_warmup_epochs=140,
            angular_momentum_warmup_epochs=140,
            correction_warmup_epochs=80,
            position_loss_weight=1.0,
            velocity_loss_weight=0.35,
            grad_clip_norm=1.0,
            compute_val_velocity_rmse=True,
            selection_metric="val_pos_rmse_km",
            force_chronological_for_derivatives=False,
            sort_train_for_derivatives=False,
            show_progress=True,
            device=device,
            distributed=is_distributed
        )

    runtime = {
        "use_gpu_profile": use_gpu_profile,
        "loader_workers": loader_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "hybrid_correction": model_cfg.hybrid_correction,
    }
    return model_cfg, train_cfg, runtime


def build_multi_stage_configs(
    dataset: dict[str, Any],
    device: str,
    loader_workers_override: int | None = None,
) -> tuple[ModelConfig, TrainConfig, TrainConfig, TrainConfig, int]:
    """Build configurations for multi-stage training (coarse, refine, physics)."""
    use_gpu_profile = device == "cuda"
    is_distributed = True if torch.cuda.device_count() > 1 else False 
    loader_workers = loader_workers_override if loader_workers_override is not None else (4 if use_gpu_profile else 0)

    # Shared model configuration
    model_cfg = ModelConfig(
        num_bodies=len(dataset["bodies"]),
        state_mode="position_only",
        backbone_type="residual",
        hidden_dim=768,
        num_layers=8,
        fourier_features=96,
        min_frequency=0.02,
        max_frequency=96.0,
        frequency_spacing="log",
        head_layers=3,
        head_hidden_dim=384,
        body_embedding_dim=64,
        interaction_layers=2,
        interaction_hidden_dim=384,
        use_layer_norm=True,
        dropout=0.0,
    )

    # Stage 1: Coarse training (data fitting only)
    coarse_cfg = TrainConfig(
        epochs=140,
        batch_size=1536,
        lr=3e-4,
        weight_decay=5e-7,
        val_fraction=0.10,
        split_mode="random",
        shuffle=True,
        train_loader_workers=loader_workers,
        pin_memory=True if use_gpu_profile else None,
        persistent_workers=True if use_gpu_profile and loader_workers > 0 else False,
        early_stopping_patience=24,
        lr_scheduler="cosine",
        min_lr=5e-7,
        nbody_loss_weight=0.0,
        physics_loss_weight=0.0,
        smoothness_loss_weight=0.0,
        energy_loss_weight=0.0,
        angular_momentum_loss_weight=0.0,
        position_loss_weight=1.0,
        velocity_loss_weight=0.0,
        grad_clip_norm=1.0,
        compute_val_velocity_rmse=False,
        selection_metric="val_pos_rmse_km",
        force_chronological_for_derivatives=False,
        sort_train_for_derivatives=False,
        show_progress=True,
        device=device,
        cuda_matmul_precision="high" if use_gpu_profile else "high",
        allow_tf32=True if use_gpu_profile else True,
        cudnn_benchmark=True if use_gpu_profile else True,
        distributed=is_distributed
    )

    # Stage 2: Refine training (add velocity supervision)
    refine_cfg = TrainConfig(
        epochs=240,
        batch_size=1536,
        lr=1.2e-4,
        weight_decay=5e-7,
        val_fraction=0.10,
        split_mode="random",
        shuffle=True,
        train_loader_workers=loader_workers,
        pin_memory=True if use_gpu_profile else None,
        persistent_workers=True if use_gpu_profile and loader_workers > 0 else False,
        early_stopping_patience=40,
        lr_scheduler="cosine",
        min_lr=5e-7,
        nbody_loss_weight=0.0,
        physics_loss_weight=0.0,
        smoothness_loss_weight=0.0,
        energy_loss_weight=0.0,
        angular_momentum_loss_weight=0.0,
        position_loss_weight=1.0,
        velocity_loss_weight=0.5,
        grad_clip_norm=1.0,
        compute_val_velocity_rmse=True,
        selection_metric="val_pos_rmse_km",
        force_chronological_for_derivatives=False,
        sort_train_for_derivatives=False,
        show_progress=True,
        device=device,
        cuda_matmul_precision="high" if use_gpu_profile else "high",
        allow_tf32=True if use_gpu_profile else True,
        cudnn_benchmark=True if use_gpu_profile else True,
        distributed=is_distributed
    )

    # Stage 3: Physics training (add n-body residual)
    physics_cfg = TrainConfig(
        epochs=80,
        batch_size=1536,
        lr=5e-5,
        weight_decay=5e-7,
        val_fraction=0.10,
        split_mode="random",
        shuffle=True,
        train_loader_workers=loader_workers,
        pin_memory=True if use_gpu_profile else None,
        persistent_workers=True if use_gpu_profile and loader_workers > 0 else False,
        early_stopping_patience=12,
        lr_scheduler="cosine",
        min_lr=5e-7,
        nbody_loss_weight=1e-6,
        adaptive_nbody_balance=True,
        nbody_target_fraction=2e-3,
        nbody_balance_beta=0.9,
        nbody_balance_max_scale=1e6,
        nbody_batch_size=48,
        nbody_start_epoch=6,
        nbody_warmup_epochs=36,
        nbody_softening_km=80_000.0,
        nbody_relative_floor_km_s2=5e-4,
        physics_loss_weight=0.0,
        smoothness_loss_weight=0.0,
        position_loss_weight=1.0,
        velocity_loss_weight=0.0,
        grad_clip_norm=1.0,
        compute_val_velocity_rmse=False,
        selection_metric="val_pos_rmse_km",
        force_chronological_for_derivatives=False,
        sort_train_for_derivatives=False,
        show_progress=True,
        device=device,
        cuda_matmul_precision="high" if use_gpu_profile else "high",
        allow_tf32=True if use_gpu_profile else True,
        cudnn_benchmark=True if use_gpu_profile else True,
        distributed=is_distributed
    )

    return model_cfg, coarse_cfg, refine_cfg, physics_cfg, loader_workers


# ============================================================================
# History Management
# ============================================================================

def save_stage_history(history: dict[str, Any], history_path: Path) -> None:
    """Save training history to JSON file for later recovery."""
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, (list, np.ndarray)):
            history_serializable[key] = [float(v) if not isinstance(v, (list, dict)) else v for v in value]
        else:
            history_serializable[key] = value
    
    history_path.write_text(json.dumps(history_serializable, indent=2, default=str))
    print(f"✓ Saved training history: {history_path}")


def load_stage_history(history_path: Path) -> dict[str, Any] | None:
    """Load training history from JSON file if it exists."""
    if not history_path.exists():
        return None
    
    try:
        history = json.loads(history_path.read_text())
        print(f"✓ Loaded training history: {history_path}")
        return history
    except Exception as e:
        print(f"⚠ Warning: Failed to load history from {history_path}: {e}")
        return None


# ============================================================================
# Plotting Functions
# ============================================================================

def save_unified_figures(plots_dir: Path, artifacts: dict[str, Any]) -> None:
    """Generate training plots for unified single-stage training."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    history = artifacts["history"]

    # Training loss
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="train data loss")
    if "train_objective_loss" in history:
        plt.plot(history["train_objective_loss"], label="train objective loss")
    plt.plot(history["val_loss"], label="val data loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("HPINN unified training history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_history.png", dpi=160)
    plt.close()

    # Validation RMSE
    plt.figure(figsize=(10, 3))
    plt.plot(history["val_pos_rmse_km"], label="val position RMSE [km]")
    if "val_vel_rmse_km_s" in history and len(history["val_vel_rmse_km_s"]) > 0:
        plt.plot(history["val_vel_rmse_km_s"], label="val velocity RMSE [km/s]")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("HPINN validation RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "validation_rmse.png", dpi=160)
    plt.close()

    # N-body diagnostics
    if "nbody_loss" in history and len(history["nbody_loss"]) > 0:
        plt.figure(figsize=(10, 3))
        nbody_raw = np.array(history["nbody_loss"], dtype=float)
        nbody_w = np.array(history["nbody_weight"], dtype=float)
        plt.plot(np.maximum(1e-16, nbody_raw), label="nbody raw")
        plt.plot(np.maximum(1e-16, nbody_raw * nbody_w), label="nbody weighted")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss contribution")
        plt.title("N-body residual diagnostics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "nbody_diagnostics.png", dpi=160)
        plt.close()

    if "correction_loss" in history and len(history["correction_loss"]) > 0:
        plt.figure(figsize=(10, 3))
        corr_raw = np.maximum(1e-16, np.array(history["correction_loss"], dtype=float))
        corr_w = np.array(history.get("correction_weight", np.zeros_like(corr_raw)), dtype=float)
        corr_amp = np.maximum(1e-16, np.array(history.get("correction_amplitude_loss", np.zeros_like(corr_raw)), dtype=float))
        corr_gain = np.maximum(1e-16, np.array(history.get("correction_gain_loss", np.zeros_like(corr_raw)), dtype=float))
        plt.plot(corr_raw, label="correction raw")
        plt.plot(np.maximum(1e-16, corr_raw * corr_w), label="correction weighted")
        plt.plot(corr_amp, label="correction amplitude")
        plt.plot(corr_gain, label="correction gain")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss contribution")
        plt.title("Hybrid correction diagnostics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "correction_diagnostics.png", dpi=160)
        plt.close()

    # Learning rate
    plt.figure(figsize=(8, 3))
    plt.plot(history["lr"])
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.title("Learning rate schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "lr_schedule.png", dpi=160)
    plt.close()


def save_multi_stage_figures(
    plots_dir: Path,
    coarse_history: dict[str, Any],
    refine_history: dict[str, Any],
    physics_history: dict[str, Any] | None,
) -> None:
    """Generate training plots for multi-stage training."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Training loss
    plt.figure(figsize=(10, 4))
    plt.plot(coarse_history["train_loss"], label="coarse train")
    plt.plot(coarse_history["val_loss"], label="coarse val")
    
    offset = len(coarse_history["train_loss"])
    plt.plot(
        range(offset, offset + len(refine_history["train_loss"])),
        refine_history["train_loss"],
        label="refine train",
    )
    plt.plot(
        range(offset, offset + len(refine_history["val_loss"])),
        refine_history["val_loss"],
        label="refine val",
    )
    
    if physics_history is not None:
        offset2 = offset + len(refine_history["train_loss"])
        plt.plot(
            range(offset2, offset2 + len(physics_history["train_loss"])),
            physics_history["train_loss"],
            label="physics train",
        )
        plt.plot(
            range(offset2, offset2 + len(physics_history["val_loss"])),
            physics_history["val_loss"],
            label="physics val",
        )
    
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Multi-stage training history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_history.png", dpi=160)
    plt.close()

    # Position RMSE
    plt.figure(figsize=(10, 3))
    plt.plot(coarse_history["val_pos_rmse_km"], label="coarse")
    
    coarse_offset = len(coarse_history["val_pos_rmse_km"])
    plt.plot(
        range(coarse_offset, coarse_offset + len(refine_history["val_pos_rmse_km"])),
        refine_history["val_pos_rmse_km"],
        label="refine",
    )
    
    if physics_history is not None:
        offset2 = coarse_offset + len(refine_history["val_pos_rmse_km"])
        plt.plot(
            range(offset2, offset2 + len(physics_history["val_pos_rmse_km"])),
            physics_history["val_pos_rmse_km"],
            label="physics",
        )
    
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("RMSE [km]")
    plt.title("Validation position RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "validation_position_rmse.png", dpi=160)
    plt.close()

    # Physics diagnostics
    if physics_history is not None and "nbody_loss" in physics_history:
        plt.figure(figsize=(10, 3))
        nbody_raw = np.array(physics_history["nbody_loss"], dtype=float)
        nbody_w = np.array(physics_history["nbody_weight"], dtype=float)
        plt.plot(nbody_raw, label="nbody raw")
        plt.plot(np.maximum(1e-16, nbody_raw * nbody_w), label="nbody weighted")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss contribution")
        plt.title("Physics stage n-body diagnostics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "physics_nbody_diagnostics.png", dpi=160)
        plt.close()


# ============================================================================
# Training Modes
# ============================================================================

def run_unified_training(args: argparse.Namespace, dataset: dict[str, Any], device: str) -> None:
    """Run single-stage unified training."""
    print("\n" + "=" * 80)
    print("UNIFIED SINGLE-STAGE TRAINING MODE")
    print("=" * 80)
    
    plots_dir = args.plots_dir or args.checkpoint_path.parent
    checkpoint_dir = Path(args.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_cfg, train_cfg, runtime = build_unified_profile(
        dataset=dataset,
        device=device,
        loader_workers_override=args.loader_workers,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        collocation_override=args.collocation_points,
    )

    print(f"\nDataset: {args.dataset_path}")
    print(f"States shape: {dataset['states'].shape}")
    print(f"Num samples: {len(dataset['times_seconds'])}")
    print(f"Device: {device}")
    print(f"GPU profile: {runtime['use_gpu_profile']}")
    print(f"Model: {model_cfg.hidden_dim}x{model_cfg.num_layers} {model_cfg.backbone_type}")
    print(f"Training: {train_cfg.epochs} epochs, batch={train_cfg.batch_size}, lr={train_cfg.lr}")
    
    print_gpu_memory_summary()
    
    print("\nStarting training...")

    unified_ckpt = checkpoint_dir / "checkpoint_unified.pt"
    artifacts = train_emulator(
        dataset,
        train_config=train_cfg,
        model_config=model_cfg,
        checkpoint_path=unified_ckpt,
        wandb_project=args.project_name
    )

    best_epoch = int(np.argmin(artifacts["history"]["val_pos_rmse_km"])) + 1
    best_rmse = float(np.min(artifacts["history"]["val_pos_rmse_km"]))
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✓ Final checkpoint: {unified_checkpoint}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val position RMSE: {best_rmse:,.2f} km")
    
    print_gpu_memory_summary()

    print("\nGenerating plots...")
    save_unified_figures(plots_dir=plots_dir, artifacts=artifacts)

    summary = {
        "training_mode": "unified",
        "device": device,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "best_val_position_rmse_km": best_rmse,
        "model_config": model_cfg.to_kwargs(),
        "train_config": vars(train_cfg),
        "runtime": runtime,
    }
    (plots_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    
    print(f"✓ Training plots: {plots_dir}")
    print(f"✓ Summary: {plots_dir / 'training_summary.json'}")


def run_multi_stage_training(args: argparse.Namespace, dataset: dict[str, Any], device: str) -> None:
    """Run three-stage training with checkpointing and resumption support."""
    print("\n" + "=" * 80)
    print("MULTI-STAGE TRAINING MODE (coarse → refine → physics)")
    print("=" * 80)
    
    plots_dir = args.plots_dir or args.checkpoint_path.parent
    checkpoint_dir = args.checkpoint_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage checkpoint paths
    coarse_ckpt = checkpoint_dir / "checkpoint_coarse.pt"
    refine_ckpt = checkpoint_dir / "checkpoint_refine.pt"
    physics_ckpt = checkpoint_dir / "checkpoint_physics.pt"
    final_checkpoint = checkpoint_dir / "final_checkpoint.pt" 
    
    # History paths for resumption
    coarse_history_path = checkpoint_dir / "history_coarse.json"
    refine_history_path = checkpoint_dir / "history_refine.json"
    physics_history_path = checkpoint_dir / "history_physics.json"

    model_cfg, coarse_cfg, refine_cfg, physics_cfg, loader_workers = build_multi_stage_configs(
        dataset=dataset,
        device=device,
        loader_workers_override=args.loader_workers,
    )

    print(f"\nDataset: {args.dataset_path}")
    print(f"States shape: {dataset['states'].shape}")
    print(f"Num samples: {len(dataset['times_seconds'])}")
    print(f"Device: {device}")
    print(f"Model: {model_cfg.hidden_dim}x{model_cfg.num_layers} {model_cfg.backbone_type}")
    print(f"DataLoader workers: {loader_workers}")
    
    print_gpu_memory_summary()

    # ========================================================================
    # STAGE 1: COARSE TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: COARSE TRAINING (position-only data fitting)")
    print("=" * 80)
    
    coarse_history = load_stage_history(coarse_history_path)
    
    if coarse_ckpt.exists() and coarse_history is not None:
        print(f"✓ Checkpoint exists: {coarse_ckpt}")
        coarse_best_epoch = int(np.argmin(coarse_history["val_pos_rmse_km"])) + 1
        coarse_best_rmse = float(np.min(coarse_history["val_pos_rmse_km"]))
        print(f"  Best epoch: {coarse_best_epoch}")
        print(f"  Best val RMSE: {coarse_best_rmse:,.2f} km")
        print("→ Skipping stage (already complete)")
    else:
        print("Training coarse stage...")
        coarse_result = train_emulator(
            dataset,
            train_config=coarse_cfg,
            model_config=model_cfg,
            checkpoint_path=coarse_ckpt,
            wandb_project=args.project_name,
            stage="coarse",
        )
        coarse_history = coarse_result["history"]
        coarse_best_epoch = int(np.argmin(coarse_history["val_pos_rmse_km"])) + 1
        coarse_best_rmse = float(np.min(coarse_history["val_pos_rmse_km"]))
        
        save_stage_history(coarse_history, coarse_history_path)
        
        print(f"✓ Checkpoint saved: {coarse_ckpt}")
        print(f"  Best epoch: {coarse_best_epoch}")
        print(f"  Best val RMSE: {coarse_best_rmse:,.2f} km")
        
        del coarse_result
        clear_gpu_memory()
        print_gpu_memory_summary()

    # ========================================================================
    # STAGE 2: REFINE TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: REFINE TRAINING (add velocity supervision)")
    print("=" * 80)
    
    refine_history = load_stage_history(refine_history_path)
    
    if refine_ckpt.exists() and refine_history is not None:
        print(f"✓ Checkpoint exists: {refine_ckpt}")
        refine_best_epoch = int(np.argmin(refine_history["val_pos_rmse_km"])) + 1
        refine_best_rmse = float(np.min(refine_history["val_pos_rmse_km"]))
        print(f"  Best epoch: {refine_best_epoch}")
        print(f"  Best val RMSE: {refine_best_rmse:,.2f} km")
        print("→ Skipping stage (already complete)")
    else:
        print(f"Training refine stage (starting from coarse checkpoint)...")
        refine_result = train_emulator(
            dataset,
            train_config=refine_cfg,
            model_config=model_cfg,
            checkpoint_path=refine_ckpt,
            initial_checkpoint_path=coarse_ckpt,
            wandb_project=args.project_name,
            stage="refine",
        )
        refine_history = refine_result["history"]
        refine_best_epoch = int(np.argmin(refine_history["val_pos_rmse_km"])) + 1
        refine_best_rmse = float(np.min(refine_history["val_pos_rmse_km"]))
        
        save_stage_history(refine_history, refine_history_path)
        
        print(f"✓ Checkpoint saved: {refine_ckpt}")
        print(f"  Best epoch: {refine_best_epoch}")
        print(f"  Best val RMSE: {refine_best_rmse:,.2f} km")
        
        del refine_result
        clear_gpu_memory()
        print_gpu_memory_summary()

    # ========================================================================
    # STAGE 3: PHYSICS TRAINING
    # ========================================================================
    physics_history = None
    physics_best_epoch = None
    physics_best_rmse = None
    
    if not args.skip_physics_stage:
        print("\n" + "=" * 80)
        print("STAGE 3: PHYSICS TRAINING (add n-body residual)")
        print("=" * 80)
        
        physics_history = load_stage_history(physics_history_path)
        
        if physics_ckpt.exists() and physics_history is not None:
            print(f"✓ Checkpoint exists: {physics_ckpt}")
            physics_best_epoch = int(np.argmin(physics_history["val_pos_rmse_km"])) + 1
            physics_best_rmse = float(np.min(physics_history["val_pos_rmse_km"]))
            print(f"  Best epoch: {physics_best_epoch}")
            print(f"  Best val RMSE: {physics_best_rmse:,.2f} km")
            print("→ Skipping stage (already complete)")
        else:
            print("Training physics stage (starting from refine checkpoint)...")
            physics_result = train_emulator(
                dataset,
                train_config=physics_cfg,
                model_config=model_cfg,
                checkpoint_path=physics_ckpt,
                initial_checkpoint_path=refine_ckpt,
                wandb_project=args.project_name,
                stage="physics",
            )
            physics_history = physics_result["history"]
            physics_best_epoch = int(np.argmin(physics_history["val_pos_rmse_km"])) + 1
            physics_best_rmse = float(np.min(physics_history["val_pos_rmse_km"]))
            
            save_stage_history(physics_history, physics_history_path)
            
            print(f"✓ Checkpoint saved: {physics_ckpt}")
            print(f"  Best epoch: {physics_best_epoch}")
            print(f"  Best val RMSE: {physics_best_rmse:,.2f} km")
            
            del physics_result
            clear_gpu_memory()
            print_gpu_memory_summary()

    # ========================================================================
    # FINAL SELECTION
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL CHECKPOINT SELECTION")
    print("=" * 80)
    
    selected_stage = "refine"
    selected_ckpt = refine_ckpt
    selected_rmse = refine_best_rmse
    
    if physics_best_rmse is not None and physics_best_rmse < selected_rmse:
        selected_stage = "physics"
        selected_ckpt = physics_ckpt
        selected_rmse = physics_best_rmse
    
    shutil.copy2(selected_ckpt, final_checkpoint)
    
    print(f"✓ Final checkpoint: {final_checkpoint}")
    print(f"  Selected stage: {selected_stage}")
    print(f"  Best val RMSE: {selected_rmse:,.2f} km")

    print("\nGenerating plots...")
    save_multi_stage_figures(
        plots_dir=plots_dir,
        coarse_history=coarse_history,
        refine_history=refine_history,
        physics_history=physics_history,
    )

    summary = {
        "training_mode": "multi",
        "device": device,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "final_checkpoint_path": str(final_checkpoint.resolve()),
        "coarse_checkpoint_path": str(coarse_ckpt.resolve()),
        "refine_checkpoint_path": str(refine_ckpt.resolve()),
        "physics_checkpoint_path": str(physics_ckpt.resolve()) if not args.skip_physics_stage else None,
        "selected_stage": selected_stage,
        "best_val_position_rmse_km": selected_rmse,
        "skip_physics_stage": args.skip_physics_stage,
        "loader_workers": loader_workers,
        "model_config": model_cfg.to_kwargs(),
        "coarse_config": vars(coarse_cfg),
        "refine_config": vars(refine_cfg),
        "physics_config": vars(physics_cfg),
    }
    (plots_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    
    print(f"✓ Training plots: {plots_dir}")
    print(f"✓ Summary: {plots_dir / 'training_summary.json'}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    print("=" * 80)
    print("SOLAR SYSTEM PINN TRAINING")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    
    dataset = load_dataset(args.dataset_path)
    device = resolve_device(args.device)
    # device = os.environ["LOCAL_RANK"]
    
    if args.training_mode == "unified":
        run_unified_training(args, dataset, device)
    elif args.training_mode == "multi":
        run_multi_stage_training(args, dataset, device)
    else:
        raise ValueError(f"Unknown training mode: {args.training_mode}")


if __name__ == "__main__":
    main()
