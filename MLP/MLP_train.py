#!/usr/bin/env python3
"""Batch-friendly training entrypoint for the MLP baseline."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Solar System MLP baseline.")
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
        "--run-stage2",
        action="store_true",
        help="Enable the optional physics fine-tuning stage from the notebook.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory where training figures and summary JSON are saved. Defaults to artifacts/.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="mlp-solar-system",
        help="WandB project name.",
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
    stage1: dict[str, Any],
    selected_artifacts: dict[str, Any],
    selected_stage: str,
    stage2: dict[str, Any] | None,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(stage1["history"]["train_loss"], label="stage1 train(data)")
    plt.plot(stage1["history"]["val_loss"], label="stage1 val(data)")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Stage 1 history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mlp_stage1_history.png", dpi=160)
    plt.close()

    if stage2 is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(stage2["history"]["train_loss"], label="stage2 train(data)")
        plt.plot(stage2["history"]["val_loss"], label="stage2 val(data)")
        plt.plot(stage2["history"]["nbody_loss"], label="stage2 nbody")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Stage 2 history")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "mlp_stage2_history.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(selected_artifacts["history"]["lr"])
    plt.title(f"Learning rate schedule ({selected_stage})")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mlp_lr_schedule.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(selected_artifacts["history"]["val_pos_rmse_km"], label="val position RMSE [km]")
    plt.plot(selected_artifacts["history"]["val_vel_rmse_km_s"], label="val velocity RMSE [km/s]")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title(f"Validation physical RMSE ({selected_stage})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mlp_validation_rmse.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    dataset = load_dataset(args.dataset_path)
    device = resolve_device(args.device)
    plots_dir = args.plots_dir or args.checkpoint_path.parent

    print("Project root:", PROJECT_ROOT)
    print("Python executable:", sys.executable)
    print("Dataset source:", dataset.get("metadata", {}).get("sample_source"))
    print("States shape:", dataset["states"].shape)
    print("Num samples:", len(dataset["times_seconds"]))
    print("Device:", device)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_ckpt = checkpoint_path.with_name("emulator_stage1.pt")
    stage2_ckpt = checkpoint_path.with_name("emulator_stage2.pt")

    stage1_cfg = TrainConfig(
        epochs=2000,
        batch_size=1536,
        lr=1e-3,
        weight_decay=1e-6,
        val_fraction=0.10,
        split_mode="random",
        shuffle=True,
        early_stopping_patience=200,
        lr_scheduler="cosine",
        min_lr=1e-6,
        nbody_loss_weight=0.0,
        physics_loss_weight=0.0,
        smoothness_loss_weight=0.0,
        position_loss_weight=1.0,
        velocity_loss_weight=1.0,
        grad_clip_norm=1.0,
        show_progress=True,
        device=device,
    )
    model_cfg = ModelConfig(
        num_bodies=len(dataset["bodies"]),
        hidden_dim=768,
        num_layers=8,
        fourier_features=256,
        min_frequency=0.02,
        max_frequency=256.0,
        frequency_spacing="log",
        head_layers=3,
        head_hidden_dim=384,
        dropout=0.0,
    )

    stage1 = train_emulator(
        dataset,
        train_config=stage1_cfg,
        model_config=model_cfg,
        checkpoint_path=stage1_ckpt,
        use_wandb=True,
        wandb_project=args.project_name,
    )
    best1_epoch = int(np.argmin(stage1["history"]["val_pos_rmse_km"])) + 1
    best1_rmse = float(np.min(stage1["history"]["val_pos_rmse_km"]))
    print("Stage1 best epoch:", best1_epoch)
    print("Stage1 best val position RMSE [km]:", f"{best1_rmse:,.2f}")

    stage2: dict[str, Any] | None = None
    selected_stage = "stage1"
    selected_ckpt = stage1_ckpt
    selected_rmse = best1_rmse
    selected_artifacts = stage1

    if args.run_stage2:
        stage2_cfg = TrainConfig(
            epochs=350,
            batch_size=384,
            lr=8e-5,
            weight_decay=1e-6,
            val_fraction=0.10,
            split_mode="random",
            shuffle=True,
            early_stopping_patience=120,
            lr_scheduler="cosine",
            min_lr=1e-7,
            nbody_loss_weight=0.01,
            nbody_warmup_epochs=120,
            nbody_softening_km=5_000.0,
            nbody_relative_floor_km_s2=1e-5,
            physics_loss_weight=5e-4,
            smoothness_loss_weight=0.0,
            position_loss_weight=1.0,
            velocity_loss_weight=1.0,
            grad_clip_norm=1.0,
            show_progress=True,
            device=device,
        )
        stage2 = train_emulator(
            dataset,
            train_config=stage2_cfg,
            model_config=model_cfg,
            checkpoint_path=stage2_ckpt,
            initial_checkpoint_path=stage1_ckpt,
            use_wandb=True,
            wandb_project=args.project_name,
        )
        best2_epoch = int(np.argmin(stage2["history"]["val_pos_rmse_km"])) + 1
        best2_rmse = float(np.min(stage2["history"]["val_pos_rmse_km"]))
        print("Stage2 best epoch:", best2_epoch)
        print("Stage2 best val position RMSE [km]:", f"{best2_rmse:,.2f}")
        if best2_rmse <= best1_rmse:
            selected_stage = "stage2"
            selected_ckpt = stage2_ckpt
            selected_rmse = best2_rmse
            selected_artifacts = stage2

    shutil.copy2(selected_ckpt, checkpoint_path)
    print("Final checkpoint:", checkpoint_path)
    print("Selected stage:", selected_stage)
    print("Best val position RMSE [km]:", f"{selected_rmse:,.2f}")

    save_figures(
        plots_dir=plots_dir,
        stage1=stage1,
        selected_artifacts=selected_artifacts,
        selected_stage=selected_stage,
        stage2=stage2,
    )
    summary = {
        "device": device,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "stage1_checkpoint_path": str(stage1_ckpt.resolve()),
        "stage2_checkpoint_path": str(stage2_ckpt.resolve()),
        "selected_stage": selected_stage,
        "best_val_position_rmse_km": selected_rmse,
        "run_stage2": bool(args.run_stage2),
        "model_config": model_cfg.to_kwargs(),
        "stage1_config": vars(stage1_cfg),
    }
    if args.run_stage2:
        summary["stage2_config"] = vars(stage2_cfg)
    (plots_dir / "mlp_train_summary.json").write_text(json.dumps(summary, indent=2))
    print("Training plots and summary saved to:", plots_dir)


if __name__ == "__main__":
    main()
