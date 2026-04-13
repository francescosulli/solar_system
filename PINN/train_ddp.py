#!/usr/bin/env python3
"""
DDP-enabled training entrypoint for multi-GPU PINN training.

This script wraps train.py with PyTorch DistributedDataParallel support.
Launch with: torchrun --nproc_per_node=2 train_ddp.py [args]
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

# DDP imports
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from solsys_emulator.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATASET_PATH, ensure_project_dirs
from solsys_emulator.de440_dataset import load_dataset
from solsys_emulator.model import ModelConfig

# Import the unified training configurations
# NOTE: We'll need to modify train_emulator to support DDP
# For now, this shows the structure


# ============================================================================
# DDP Utilities
# ============================================================================

# def setup_ddp():
#     """Initialize the distributed process group."""
#     if not dist.is_available():
#         raise RuntimeError("PyTorch distributed not available")
    
#     # Initialize process group
#     dist.init_process_group(backend="nccl")
    
#     # Get rank and world size
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
#     # Set device
#     torch.cuda.set_device(local_rank)
    
#     return rank, world_size, local_rank

def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    return local_rank, world_size, device

def cleanup_ddp():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


# ============================================================================
# DDP-Aware Training Configuration
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DDP training."""
    parser = argparse.ArgumentParser(
        description="DDP-enabled PINN training for multi-GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Launch with torchrun:
  # 2 GPUs on single node
  torchrun --nproc_per_node=2 train_ddp.py --training-mode unified
  
  # 4 GPUs across 2 nodes (run on each node)
  torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=<IP> train_ddp.py
  
Environment variables set by torchrun:
  RANK: Global rank of this process
  LOCAL_RANK: Local rank on this node
  WORLD_SIZE: Total number of processes
  MASTER_ADDR: Address of rank 0
  MASTER_PORT: Port for communication
        """
    )
    
    parser.add_argument(
        "--training-mode",
        type=str,
        default="unified",
        choices=("unified", "multi"),
        help="Training mode: 'unified' or 'multi-stage'",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-GPU batch size")
    parser.add_argument("--collocation-points", type=int, default=None)
    parser.add_argument("--skip-physics-stage", action="store_true")
    
    # DDP-specific arguments
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (for simulating larger batch sizes)",
    )
    parser.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help="Enable DDP's find_unused_parameters (slower but handles dynamic graphs)",
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-rank logs (default: logs/)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (enables W&B logging)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (username or team)",
    )
    
    return parser.parse_args()


def get_ddp_batch_size(base_batch_size: int, world_size: int) -> int:
    """
    Calculate per-GPU batch size.
    
    In DDP, each GPU processes its own batch, so we typically want to keep
    the same per-GPU batch size, not divide it by world_size.
    """
    return base_batch_size


# ============================================================================
# Logging Setup
# ============================================================================

class MultiGPULogger:
    """
    Logger that handles per-rank logging and aggregates metrics across GPUs.
    
    Features:
    - Per-rank log files for debugging
    - Synchronized metric logging to main process
    - Optional W&B integration
    - GPU memory tracking
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        log_dir: Path | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        
        # Set up per-rank log file
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"rank_{rank}.log"
        
        # W&B setup (only on main process)
        self.use_wandb = wandb_project is not None and self.is_main
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config={
                        "world_size": world_size,
                        "distributed": True,
                    },
                )
                self.wandb = wandb
            except ImportError:
                print_rank0("Warning: wandb requested but not installed")
                self.use_wandb = False
    
    def log(self, message: str, to_file: bool = True, to_console: bool = True):
        """Log message to file and optionally console."""
        if to_file:
            with open(self.log_file, "a") as f:
                f.write(f"[Rank {self.rank}] {message}\n")
        
        if to_console:
            print(f"[Rank {self.rank}] {message}")
    
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
        aggregate: bool = True,
    ):
        """
        Log metrics, optionally aggregating across GPUs.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/epoch
            prefix: Prefix for metric names (e.g., "train/", "val/")
            aggregate: If True, average metrics across all GPUs
        """
        if aggregate and self.world_size > 1:
            # Aggregate metrics across GPUs
            aggregated = {}
            for key, value in metrics.items():
                tensor = torch.tensor(value, device=f"cuda:{self.rank}")
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                aggregated[key] = (tensor / self.world_size).item()
            metrics = aggregated
        
        # Log to file
        metrics_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        self.log(f"Step {step}: {prefix}{metrics_str}")
        
        # Log to W&B (main process only)
        if self.use_wandb and self.is_main:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            self.wandb.log(wandb_metrics, step=step)
    
    def log_gpu_memory(self, step: int):
        """Log GPU memory usage for this rank."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            self.log(
                f"Step {step} GPU Memory: "
                f"allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB, "
                f"max_allocated={max_allocated:.2f}GB"
            )
    
    def close(self):
        """Clean up logger resources."""
        if self.use_wandb:
            self.wandb.finish()


# ============================================================================
# DDP Training Wrapper
# ============================================================================

def train_emulator_ddp(
    dataset: dict[str, Any],
    train_config: Any,  # TrainConfig
    model_config: ModelConfig,
    checkpoint_path: Path,
    initial_checkpoint_path: Path | None = None,
    logger: MultiGPULogger | None = None,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    find_unused_parameters: bool = False,
) -> dict[str, Any]:
    """
    DDP-enabled wrapper around train_emulator.
    
    This function needs to be integrated into solsys_emulator/train.py
    or we need to modify that file to support DDP natively.
    
    Key modifications needed:
    1. Wrap model with DDP
    2. Use DistributedSampler for DataLoader
    3. Synchronize metrics across ranks
    4. Save checkpoints only from rank 0
    5. Handle gradient accumulation correctly
    """
    
    # This is a placeholder showing the structure
    # The actual implementation would modify solsys_emulator/train.py
    
    raise NotImplementedError(
        "DDP training requires modifications to solsys_emulator/train.py. "
        "See the DDP implementation guide in DDP_IMPLEMENTATION.md"
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main DDP training entry point."""
    args = parse_args()
    
    # Initialize DDP
    rank, world_size, local_rank = setup_ddp()
    
    # Set up logger
    logger = MultiGPULogger(
        rank=rank,
        world_size=world_size,
        log_dir=args.log_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    
    try:
        logger.log("=" * 80)
        logger.log("DDP TRAINING INITIALIZATION")
        logger.log("=" * 80)
        logger.log(f"Rank: {rank}/{world_size}")
        logger.log(f"Local rank: {local_rank}")
        logger.log(f"Device: cuda:{local_rank}")
        logger.log(f"Training mode: {args.training_mode}")
        
        # Ensure all ranks are ready
        barrier()
        
        # Load dataset (all ranks load it)
        print_rank0("Loading dataset...")
        dataset = load_dataset(args.dataset_path)
        print_rank0(f"Dataset loaded: {dataset['states'].shape}")
        
        barrier()
        
        # NOTE: This is where you would call the DDP-enabled training
        # For now, this raises NotImplementedError
        print_rank0("\nTo complete DDP implementation:")
        print_rank0("1. Modify solsys_emulator/train.py to support DDP")
        print_rank0("2. See DDP_IMPLEMENTATION.md for detailed guide")
        print_rank0("3. Key changes: wrap model with DDP, use DistributedSampler")
        
        raise NotImplementedError("See DDP_IMPLEMENTATION.md for implementation guide")
        
    except Exception as e:
        logger.log(f"ERROR: {e}", to_console=True)
        raise
    finally:
        logger.close()
        cleanup_ddp()


if __name__ == "__main__":
    main()
