"""Distributed logging utilities for multi-GPU training."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


class DistributedLogger:
    """Logger that handles per-rank file logging and rank-0 console output."""
    
    def __init__(
        self,
        name: str,
        log_dir: Path,
        rank: int = 0,
        world_size: int = 1,
        log_level: int = logging.INFO,
    ):
        self.name = name
        self.rank = rank
        self.world_size = world_size
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"{name}_rank{rank}")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler (all ranks)
        log_file = self.log_dir / f"rank_{rank}.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            fmt='[%(asctime)s][Rank %(rank)d][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (rank 0 only)
        if rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                fmt='[%(asctime)s][%(levelname)s] %(message)s',
                datefmt='%H:%M:%S',
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Add rank to log records
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.rank = rank
            return record
        logging.setLogRecordFactory(record_factory)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, **kwargs)
    
    def log_gpu_memory(self, step: int):
        """Log GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        device_id = self.rank % torch.cuda.device_count()
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
        
        self.info(
            f"Step {step} | GPU Memory: "
            f"allocated={allocated:.2f}GB, "
            f"reserved={reserved:.2f}GB, "
            f"peak={max_allocated:.2f}GB"
        )
    
    def log_gradients(self, model: torch.nn.Module, step: int):
        """Log gradient statistics."""
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                max_grad = max(max_grad, grad_norm)
                min_grad = min(min_grad, grad_norm)
        
        total_norm = total_norm ** 0.5
        
        self.debug(
            f"Step {step} | Gradients: "
            f"total_norm={total_norm:.6f}, "
            f"max={max_grad:.6f}, "
            f"min={min_grad:.6f}"
        )
    
    def log_batch_stats(self, batch_data: torch.Tensor, step: int):
        """Log batch statistics."""
        self.debug(
            f"Step {step} | Batch: "
            f"shape={batch_data.shape}, "
            f"mean={batch_data.mean().item():.6f}, "
            f"std={batch_data.std().item():.6f}, "
            f"min={batch_data.min().item():.6f}, "
            f"max={batch_data.max().item():.6f}"
        )
    
    def sync_and_log(self, msg: str, level: str = 'info'):
        """Synchronize all ranks and log message."""
        if dist.is_initialized():
            dist.barrier()
        
        getattr(self, level)(msg)

