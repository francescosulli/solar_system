#!/usr/bin/env python3
"""Batch-friendly training entrypoint for the PINN model."""

from __future__ import annotations

import argparse
import json
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Solar System PINN with a single unified run.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="Input dataset .npz path.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Final checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. 'auto' selects cuda when available.",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=None,
        help="Override DataLoader workers. By default: 8 on cuda, 0 on cpu.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs. Default depends on device profile.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size. Default depends on device profile.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps. Useful to lower peak GPU memory.",
    )
    parser.add_argument(
        "--collocation-points",
        type=int,
        default=None,
        help="Override collocation points per batch for n-body residual.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory where training figures and summary JSON are saved. Defaults to artifacts/.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
    return device_arg


def build_profile(
    dataset: dict[str, Any],
    device: str,
    loader_workers_override: int | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    grad_accum_override: int | None = None,
    collocation_override: int | None = None,
) -> tuple[ModelConfig, TrainConfig, dict[str, Any]]:
    use_gpu_profile = device == "cuda"
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
            use_layer_norm=True,
            dropout=0.0,
        )
        train_cfg = TrainConfig(
            epochs=epochs_override or 1400,
            batch_size=batch_size_override or 768,
            gradient_accumulation_steps=grad_accum_override or 2,
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
            energy_start_epoch=40,
            angular_momentum_start_epoch=40,
            energy_warmup_epochs=320,
            angular_momentum_warmup_epochs=320,
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
            use_layer_norm=False,
            dropout=0.0,
        )
        train_cfg = TrainConfig(
            epochs=epochs_override or 600,
            batch_size=batch_size_override or 384,
            gradient_accumulation_steps=grad_accum_override or 1,
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
            energy_start_epoch=20,
            angular_momentum_start_epoch=20,
            energy_warmup_epochs=140,
            angular_momentum_warmup_epochs=140,
            position_loss_weight=1.0,
            velocity_loss_weight=0.35,
            grad_clip_norm=1.0,
            compute_val_velocity_rmse=True,
            selection_metric="val_pos_rmse_km",
            force_chronological_for_derivatives=False,
            sort_train_for_derivatives=False,
            show_progress=True,
            device=device,
        )

    runtime = {
        "use_gpu_profile": use_gpu_profile,
        "loader_workers": loader_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "effective_batch_size": int(train_cfg.batch_size) * int(train_cfg.gradient_accumulation_steps),
    }
    return model_cfg, train_cfg, runtime


def save_figures(plots_dir: Path, artifacts: dict[str, Any]) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    history = artifacts["history"]

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="train data loss")
    if "train_objective_loss" in history:
        plt.plot(history["train_objective_loss"], label="train objective loss")
    plt.plot(history["val_loss"], label="val data loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("PINN unified training history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pinn_unified_training_history.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(history["val_pos_rmse_km"], label="val position RMSE [km]")
    plt.plot(history["val_vel_rmse_km_s"], label="val velocity RMSE [km/s]")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("PINN unified validation RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pinn_unified_validation_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 3))
    nbody_raw = np.array(history["nbody_loss"], dtype=float)
    nbody_w = np.array(history["nbody_weight"], dtype=float)
    plt.plot(np.maximum(1e-16, nbody_raw), label="nbody raw")
    plt.plot(np.maximum(1e-16, nbody_raw * nbody_w), label="nbody weighted")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss contribution scale")
    plt.title("PINN n-body residual diagnostics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pinn_unified_nbody_diagnostics.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(history["lr"])
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("PINN learning-rate schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pinn_unified_lr_schedule.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    dataset = load_dataset(args.dataset_path)
    device = resolve_device(args.device)
    plots_dir = args.plots_dir or args.checkpoint_path.parent
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg, train_cfg, runtime = build_profile(
        dataset=dataset,
        device=device,
        loader_workers_override=args.loader_workers,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        grad_accum_override=args.grad_accum_steps,
        collocation_override=args.collocation_points,
    )

    print("Project root:", PROJECT_ROOT)
    print("Python executable:", sys.executable)
    print("Dataset source:", dataset.get("metadata", {}).get("sample_source"))
    print("States shape:", dataset["states"].shape)
    print("Num samples:", len(dataset["times_seconds"]))
    print("Device:", device)
    print("GPU-heavy profile:", runtime["use_gpu_profile"])
    print("Effective batch size:", runtime["effective_batch_size"])
    print("Model config:", model_cfg)
    print("Train config:", train_cfg)
    print("Note: for position_only PINN, the active physics term is n-body residual; physics/smoothness losses stay disabled by design.")

    artifacts = train_emulator(
        dataset,
        train_config=train_cfg,
        model_config=model_cfg,
        checkpoint_path=checkpoint_path,
    )

    best_epoch = int(np.argmin(artifacts["history"]["val_pos_rmse_km"])) + 1
    best_rmse = float(np.min(artifacts["history"]["val_pos_rmse_km"]))
    print("Final PINN checkpoint:", checkpoint_path)
    print("Best epoch:", best_epoch)
    print("Best val position RMSE [km]:", f"{best_rmse:,.2f}")

    save_figures(plots_dir=plots_dir, artifacts=artifacts)
    summary = {
        "device": device,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "best_val_position_rmse_km": best_rmse,
        "model_config": model_cfg.to_kwargs(),
        "train_config": vars(train_cfg),
        "runtime": runtime,
    }
    (plots_dir / "pinn_train_summary.json").write_text(json.dumps(summary, indent=2))
    print("Training plots and summary saved to:", plots_dir)


if __name__ == "__main__":
    main()
