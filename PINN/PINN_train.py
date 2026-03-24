#!/usr/bin/env python3
"""Batch-friendly training entrypoint for the PINN model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
import gc         # garbage collect library

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
    parser = argparse.ArgumentParser(description="Train the Solar System PINN.")
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
        "--skip-physics-stage",
        action="store_true",
        help="Skip the final n-body PINN fine-tuning stage.",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=None,
        help="Override DataLoader workers. By default: 4 on cuda, 0 on cpu.",
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


def save_figures(
    plots_dir: Path,
    coarse: dict[str, Any],
    refine: dict[str, Any],
    physics: dict[str, Any] | None,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(coarse["history"]["train_loss"], label="coarse train(data)")
    plt.plot(coarse["history"]["val_loss"], label="coarse val(data)")
    offset = len(coarse["history"]["train_loss"])
    plt.plot(
        range(offset, offset + len(refine["history"]["train_loss"])),
        refine["history"]["train_loss"],
        label="refine train(data)",
    )
    plt.plot(
        range(offset, offset + len(refine["history"]["val_loss"])),
        refine["history"]["val_loss"],
        label="refine val(data)",
    )
    if physics is not None:
        offset2 = offset + len(refine["history"]["train_loss"])
        plt.plot(
            range(offset2, offset2 + len(physics["history"]["train_loss"])),
            physics["history"]["train_loss"],
            label="physics train(data)",
        )
        plt.plot(
            range(offset2, offset2 + len(physics["history"]["val_loss"])),
            physics["history"]["val_loss"],
            label="physics val(data)",
        )
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("PINN training history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pinn_training_history.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(coarse["history"]["val_pos_rmse_km"], label="coarse val position RMSE [km]")
    coarse_offset = len(coarse["history"]["val_pos_rmse_km"])
    plt.plot(
        range(coarse_offset, coarse_offset + len(refine["history"]["val_pos_rmse_km"])),
        refine["history"]["val_pos_rmse_km"],
        label="refine val position RMSE [km]",
    )
    if physics is not None:
        offset2 = coarse_offset + len(refine["history"]["val_pos_rmse_km"])
        plt.plot(
            range(offset2, offset2 + len(physics["history"]["val_pos_rmse_km"])),
            physics["history"]["val_pos_rmse_km"],
            label="physics val position RMSE [km]",
        )
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("PINN validation position RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pinn_validation_position_rmse.png", dpi=160)
    plt.close()

    if physics is not None:
        plt.figure(figsize=(10, 3))
        nbody_raw = np.array(physics["history"]["nbody_loss"], dtype=float)
        nbody_w = np.array(physics["history"]["nbody_weight"], dtype=float)
        plt.plot(nbody_raw, label="physics nbody raw")
        plt.plot(np.maximum(1e-16, nbody_raw * nbody_w), label="physics nbody weighted")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss contribution scale")
        plt.title("Physics-stage contribution diagnostics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "pinn_physics_contributions.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    dataset = load_dataset(args.dataset_path)
    device = resolve_device(args.device)
    use_gpu_profile = device == "cuda"
    loader_workers = args.loader_workers if args.loader_workers is not None else (4 if use_gpu_profile else 0)
    pin_memory = True if use_gpu_profile else None
    persistent_workers = True if use_gpu_profile and loader_workers > 0 else False
    plots_dir = args.plots_dir or args.checkpoint_path.parent

    print("Project root:", PROJECT_ROOT)
    print("Python executable:", sys.executable)
    print("Dataset source:", dataset.get("metadata", {}).get("sample_source"))
    print("States shape:", dataset["states"].shape)
    print("Num samples:", len(dataset["times_seconds"]))
    print("Device:", device)
    print("GPU-heavy profile:", use_gpu_profile)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    coarse_ckpt = checkpoint_path.with_name("emulator_pinn_coarse.pt")
    refine_ckpt = checkpoint_path.with_name("emulator_pinn_refine.pt")
    physics_ckpt = checkpoint_path.with_name("emulator_pinn_physics.pt")

    if use_gpu_profile:
        model_cfg = ModelConfig(
            num_bodies=len(dataset["bodies"]),
            state_mode="position_only",
            backbone_type="residual",
            hidden_dim=640,
            num_layers=8,
            fourier_features=64,
            min_frequency=0.05,
            max_frequency=64.0,
            frequency_spacing="log",
            head_layers=3,
            head_hidden_dim=256,
            body_embedding_dim=32,
            use_layer_norm=True,
            dropout=0.0,
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
            use_layer_norm=False,
            dropout=0.0,
        )
    print("Model config:", model_cfg)

    if use_gpu_profile:
        coarse_cfg = TrainConfig(
            epochs=1000,
            batch_size=2048,
            lr=3e-4,
            weight_decay=1e-6,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
            train_loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            early_stopping_patience=180,
            lr_scheduler="cosine",
            min_lr=5e-6,
            nbody_loss_weight=0.0,
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
        )
        refine_cfg = TrainConfig(
            epochs=320,
            batch_size=1024,
            lr=5e-5,
            weight_decay=1e-6,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
            train_loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            early_stopping_patience=90,
            lr_scheduler="cosine",
            min_lr=1e-6,
            nbody_loss_weight=0.0,
            physics_loss_weight=0.0,
            smoothness_loss_weight=0.0,
            position_loss_weight=1.0,
            velocity_loss_weight=0.6,
            grad_clip_norm=1.0,
            compute_val_velocity_rmse=True,
            selection_metric="val_pos_rmse_km",
            force_chronological_for_derivatives=False,
            sort_train_for_derivatives=False,
            show_progress=True,
            device=device,
        )
        physics_cfg = TrainConfig(
            epochs=160,
            batch_size=768,
            lr=5e-6,
            weight_decay=1e-6,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
            train_loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            early_stopping_patience=40,
            lr_scheduler="cosine",
            min_lr=1e-6,
            nbody_loss_weight=2e-6,
            adaptive_nbody_balance=True,
            nbody_target_fraction=5e-3,
            nbody_balance_beta=0.95,
            nbody_balance_max_scale=1e7,
            nbody_collocation_points=256,
            nbody_start_epoch=8,
            nbody_warmup_epochs=80,
            nbody_softening_km=80_000.0,
            nbody_relative_floor_km_s2=5e-4,
            physics_loss_weight=0.0,
            smoothness_loss_weight=0.0,
            position_loss_weight=1.0,
            velocity_loss_weight=0.2,
            grad_clip_norm=1.0,
            compute_val_velocity_rmse=True,
            selection_metric="val_pos_rmse_km",
            force_chronological_for_derivatives=False,
            sort_train_for_derivatives=False,
            show_progress=True,
            device=device,
        )
    else:
        coarse_cfg = TrainConfig(
            epochs=450,
            batch_size=512,
            lr=3e-4,
            weight_decay=1e-6,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
            early_stopping_patience=100,
            lr_scheduler="cosine",
            min_lr=1e-6,
            nbody_loss_weight=0.0,
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
        )
        refine_cfg = TrainConfig(
            epochs=220,
            batch_size=384,
            lr=3e-5,
            weight_decay=1e-6,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
            early_stopping_patience=60,
            lr_scheduler="cosine",
            min_lr=1e-6,
            nbody_loss_weight=0.0,
            physics_loss_weight=0.0,
            smoothness_loss_weight=0.0,
            position_loss_weight=1.0,
            velocity_loss_weight=0.4,
            grad_clip_norm=1.0,
            compute_val_velocity_rmse=True,
            selection_metric="val_pos_rmse_km",
            force_chronological_for_derivatives=False,
            sort_train_for_derivatives=False,
            show_progress=True,
            device=device,
        )
        physics_cfg = TrainConfig(
            epochs=60,
            batch_size=384,
            lr=2e-6,
            weight_decay=1e-6,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
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
        )

      
    if not any("emulator_pinn_coarse.pt" in f.name for f in Path(checkpoint_path).iterdir()):
        coarse = train_emulator(
            dataset,
            train_config=coarse_cfg,
            model_config=model_cfg,
            checkpoint_path=coarse_ckpt,
        )
        coarse_best_epoch = int(np.argmin(coarse["history"]["val_pos_rmse_km"])) + 1
        coarse_best_rmse = float(np.min(coarse["history"]["val_pos_rmse_km"]))
        print("Coarse checkpoint:", coarse_ckpt)
        print("Coarse best epoch:", coarse_best_epoch)
        print("Coarse best val position RMSE [km]:", f"{coarse_best_rmse:,.2f}")
        del coarse # make sure to delete models from the GPU, otherwise V100s will overfill and crash with CUDA OOM.

        gc.collect()
        torch.cuda.empty_cache() 

    else:
        print("Checkpoint for coarse was provided: skipping coarse stage...")
    
    
    if not any("emulator_pinn_refine.pt" in f.name for f in Path(checkpoint_path).iterdir()):
        refine = train_emulator(
            dataset,
            train_config=refine_cfg,
            model_config=model_cfg,
            checkpoint_path=refine_ckpt,
            initial_checkpoint_path=coarse_ckpt,
        )
        refine_best_epoch = int(np.argmin(refine["history"]["val_pos_rmse_km"])) + 1
        refine_best_rmse = float(np.min(refine["history"]["val_pos_rmse_km"]))
        print("Refine checkpoint:", refine_ckpt)
        print("Refine best epoch:", refine_best_epoch)
        print("Refine best val position RMSE [km]:", f"{refine_best_rmse:,.2f}")
        
        del refine

        gc.collect()
        torch.cuda.empty_cache() 

    else:
        print("Checkpoint for refine was provided: skipping refine stage...")

    physics: dict[str, Any] | None = None
    physics_best_epoch: int | None = None
    physics_best_rmse: float | None = None

    if not args.skip_physics_stage:
        physics = train_emulator(
            dataset,
            train_config=physics_cfg,
            model_config=model_cfg,
            checkpoint_path=physics_ckpt,
            initial_checkpoint_path=refine_ckpt,
        )
        physics_best_epoch = int(np.argmin(physics["history"]["val_pos_rmse_km"])) + 1
        physics_best_rmse = float(np.min(physics["history"]["val_pos_rmse_km"]))
        print("Physics checkpoint:", physics_ckpt)
        print("Physics best epoch:", physics_best_epoch)
        print("Physics best val position RMSE [km]:", f"{physics_best_rmse:,.2f}")

    selected_stage = "refine"
    selected_ckpt = refine_ckpt
    selected_rmse = refine_best_rmse
    if physics is not None and physics_best_rmse is not None and physics_best_rmse < selected_rmse:
        selected_stage = "physics"
        selected_ckpt = physics_ckpt
        selected_rmse = physics_best_rmse
    
    del physics

    shutil.copy2(selected_ckpt, checkpoint_path)
    print("Final PINN checkpoint:", checkpoint_path)
    print("Selected stage:", selected_stage)
    print("Best val position RMSE [km]:", f"{selected_rmse:,.2f}")

    save_figures(
        plots_dir=plots_dir,
        coarse=coarse,
        refine=refine,
        physics=physics,
    )
    summary = {
        "device": device,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "coarse_checkpoint_path": str(coarse_ckpt.resolve()),
        "refine_checkpoint_path": str(refine_ckpt.resolve()),
        "physics_checkpoint_path": str(physics_ckpt.resolve()),
        "selected_stage": selected_stage,
        "best_val_position_rmse_km": selected_rmse,
        "skip_physics_stage": bool(args.skip_physics_stage),
        "loader_workers": loader_workers,
        "model_config": model_cfg.to_kwargs(),
        "coarse_config": vars(coarse_cfg),
        "refine_config": vars(refine_cfg),
        "physics_config": vars(physics_cfg),
    }
    (plots_dir / "pinn_train_summary.json").write_text(json.dumps(summary, indent=2))
    print("Training plots and summary saved to:", plots_dir)


if __name__ == "__main__":
    main()
