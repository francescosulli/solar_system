#!/usr/bin/env python3
"""
Solar System PINN Emulator — unified pipeline script.

Stages:
  1. Setup & sanity checks
  2. Build dataset from DE440 kernel
  3. Train emulator (Stage 1 data fit + optional Stage 2 physics fine-tuning)
  4. Inference & trajectory comparison against DE440
  5. Gravity-field API demo

Run:
    python run_pipeline.py [--help]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for HPC nodes
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv

# ---------------------------------------------------------------------------
# Project-root resolution (works when run from any working directory)
# ---------------------------------------------------------------------------
_CANDIDATES = [
    Path(__file__).resolve().parent,
    Path.cwd().resolve(),
    Path.cwd().resolve().parent,
    Path.cwd().resolve().parent.parent,
]
_PROJECT_ROOT = next(
    (p for p in _CANDIDATES if (p / "src" / "solsys_emulator").exists()), None
)
if _PROJECT_ROOT is None:
    sys.exit(
        "ERROR: Cannot locate project root containing src/solsys_emulator. "
        "Place run_pipeline.py inside the project directory."
    )

_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Package imports (available after sys.path is fixed)
# ---------------------------------------------------------------------------
from solsys_emulator.config import (
    DEFAULT_BODIES,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_KERNEL_PATH,
    FRAME_INTERNAL,
    FRAME_ORIGIN,
    TIME_SCALE,
    UNIT_SYSTEM,
    ensure_project_dirs,
)
from solsys_emulator.constants import get_mu
from solsys_emulator.de440_dataset import (
    build_dataset,
    find_local_kernel,
    load_dataset,
    load_kernel,
    save_dataset,
    sample_states,
)
from solsys_emulator.gravity_field import acceleration, potential
from solsys_emulator.inference import EphemerisEmulator
from solsys_emulator.model import ModelConfig
from solsys_emulator.time_frames import build_time_grid
from solsys_emulator.train import TrainConfig, load_checkpoint, train_emulator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# CLI
# ===========================================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Solar System PINN emulator — end-to-end pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    p.add_argument("--start", default="2010-01-01T00:00:00", help="Dataset start (ISO-8601 TDB)")
    p.add_argument("--end", default="2030-01-01T00:00:00", help="Dataset end (ISO-8601 TDB)")
    p.add_argument("--step-hours", type=float, default=3.0, help="Sampling step in hours")
    p.add_argument(
        "--kernel", default=None,
        help="Path to DE440/DE441 BSP kernel. Auto-detected from data/ if omitted.",
    )
    p.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH), help="Output .npz path")
    p.add_argument(
        "--skip-dataset", action="store_true",
        help="Skip dataset generation and load an existing .npz",
    )

    # # Training
    # p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="Checkpoint .pt path")
    # p.add_argument("--epochs-stage1", type=int, default=1000)
    # p.add_argument("--epochs-stage2", type=int, default=350)
    # p.add_argument("--batch-size", type=int, default=384)
    # p.add_argument("--hidden-dim", type=int, default=384)
    # p.add_argument("--num-layers", type=int, default=6)
    # p.add_argument("--fourier-features", type=int, default=24)
    # p.add_argument("--do-stage2", action="store_true", help="Enable physics fine-tuning Stage 2")
    # p.add_argument("--skip-train", action="store_true", help="Skip training; load existing checkpoint")
    # p.add_argument("--device", default=None, help="'cuda', 'cpu', or None (auto)")

    # # Inference
    # p.add_argument("--infer-start", default="2025-01-01T00:00:00")
    # p.add_argument("--infer-end", default="2026-01-01T00:00:00")
    # p.add_argument("--infer-step-hours", type=float, default=6.0)

    # # Output
    # p.add_argument("--output-dir", default=str(_PROJECT_ROOT / "artifacts"), help="Directory for plots")

    return p.parse_args()


# ===========================================================================
# Stage 1 — Setup & checks
# ===========================================================================
def stage_setup(args: argparse.Namespace) -> None:
    log.info("=== Stage 0: Setup & dependency check ===")
    ensure_project_dirs()

    log.info("Project root : %s", _PROJECT_ROOT)
    log.info("Python       : %s", sys.executable)
    log.info("Frame        : %s | Origin: %s", FRAME_INTERNAL, FRAME_ORIGIN)
    log.info("Time scale   : %s", TIME_SCALE)
    log.info("Unit system  : %s", UNIT_SYSTEM)
    log.info("Bodies       : %s", DEFAULT_BODIES)

    import importlib
    required_deps = [
        "numpy", "scipy", "astropy", "jplephem",
        "torch", "tqdm", "plotly", "matplotlib",
    ]
    missing = [d for d in required_deps if importlib.util.find_spec(d) is None]
    if missing:
        log.warning("Missing optional packages: %s", missing)
    else:
        log.info("All required packages found.")

    kernel = find_local_kernel(
        [args.kernel] if args.kernel else [DEFAULT_KERNEL_PATH]
    )
    if kernel:
        log.info("DE440 kernel : %s", kernel)
    else:
        log.warning("No DE440/DE441 kernel found in data/. Trying to download...")
        try:
            from solsys_emulator.de440_dataset import download_kernel

            log.warning("No DE440/DE441 kernel found in data/. Trying to download kernel")
            download_kernel()
            log.info("Kernel successfully downloaded.")
            kernel = find_local_kernel(
                [args.kernel] if args.kernel else [DEFAULT_KERNEL_PATH]
            )
            if kernel:
                log.info("Downloaded DE440 kernel : %s", kernel)
        except:
            log.warning("No DE440/DE441 kernel downloaded. Data generation will fail.")

# ===========================================================================
# Stage 2 — Build dataset
# ===========================================================================
def stage_build_dataset(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset_path)

    if args.skip_dataset:
        log.info("=== Stage 1: Loading existing dataset from %s ===", dataset_path)
        dataset = load_dataset(dataset_path)
        log.info("States shape  : %s", dataset["states"].shape)
        log.info("Num samples   : %d", len(dataset["times_seconds"]))
        log.info("Source        : %s", dataset["metadata"].get("sample_source"))
        return dataset

    log.info("=== Stage 1: Building dataset ===")
    step_seconds = args.step_hours * 3600.0

    kernel_candidates = [args.kernel] if args.kernel else [DEFAULT_KERNEL_PATH]
    kernel_path = find_local_kernel(kernel_candidates)
    if kernel_path is None:
        sys.exit(
            "ERROR: wasn't able to download DE440/DE441 kernel. "
            "ERROR: No DE440/DE441 kernel found. "
            "Place de440.bsp in data/ or pass --kernel <path>."
        )

    log.info("Kernel path   : %s", kernel_path)
    log.info("Date range    : %s → %s", args.start, args.end)
    log.info("Step          : %.1f h", args.step_hours)

    dataset = build_dataset(
        start_time=args.start,
        end_time=args.end,
        step=step_seconds,
        bodies=DEFAULT_BODIES,
        kernel_path=kernel_path,
    )

    log.info("States shape  : %s", dataset["states"].shape)
    log.info("Num samples   : %d", len(dataset["times_seconds"]))
    log.info("Frame         : %s | Timescale: %s", dataset["metadata"]["frame"], dataset["metadata"]["timescale"])

    save_path = save_dataset(dataset_path, dataset)
    log.info("Dataset saved : %s", save_path)
    return dataset
def main():
    args = _parse_args()
    stage_setup(args)
    stage_build_dataset(args)

if __name__ == "__main__":
    main()
