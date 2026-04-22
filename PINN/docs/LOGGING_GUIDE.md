# Multi-GPU Logging and Debugging Guide

Complete guide for setting up professional logging, monitoring, and debugging for distributed PINN training.

## Table of Contents
1. [Logging Architecture](#logging-architecture)
2. [TensorBoard Setup](#tensorboard-setup)
3. [Weights & Biases Setup](#weights--biases-setup)
4. [Per-Rank Logging](#per-rank-logging)
5. [GPU Memory Profiling](#gpu-memory-profiling)
6. [Debugging Workflows](#debugging-workflows)
7. [Production Monitoring](#production-monitoring)

---

## Logging Architecture

### Three-Tier Logging System

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Per-Rank Logs (local files)                        │
│  - Detailed debug info for each GPU                        │
│  - Gradient norms, memory usage, timing                    │
│  - Location: logs/rank_0.log, logs/rank_1.log, ...       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Aggregated Metrics (TensorBoard/W&B)              │
│  - Training curves, validation metrics                     │
│  - System metrics (GPU util, memory)                       │
│  - Only from rank 0 to avoid duplicates                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Alerts & Dashboards (optional)                    │
│  - Slack/email alerts for failures                         │
│  - Real-time monitoring dashboards                         │
│  - Integration with cluster monitoring                     │
└─────────────────────────────────────────────────────────────┘
```

---

## TensorBoard Setup

### Installation

```bash
pip install tensorboard torch-tb-profiler
```

### Implementation

Add to `src/solsys_emulator/train.py`:

```python
from torch.utils.tensorboard import SummaryWriter
import torch.profiler as profiler

def train_emulator(..., enable_tensorboard=True, log_dir="runs"):
    """Train with TensorBoard logging."""
    
    # Initialize writer (only on rank 0)
    writer = None
    if rank == 0 and enable_tensorboard:
        from datetime import datetime
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu{world_size}"
        writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")
        
        # Log hyperparameters
        hparams = {
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'hidden_dim': mcfg.hidden_dim,
            'num_layers': mcfg.num_layers,
            'world_size': world_size,
        }
        writer.add_hparams(hparams, {})
        
        # Log model graph
        dummy_input = torch.randn(1, ..., device=device)
        writer.add_graph(model, dummy_input)
    
    for epoch_idx in epoch_iter:
        # ... training loop ...
        
        if rank == 0 and writer is not None:
            # Scalars
            writer.add_scalar('Loss/train', epoch_train_loss, epoch_idx)
            writer.add_scalar('Loss/val', epoch_val_loss, epoch_idx)
            writer.add_scalar('RMSE/position', val_pos_rmse, epoch_idx)
            writer.add_scalar('RMSE/velocity', val_vel_rmse, epoch_idx)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch_idx)
            
            # Physics losses
            writer.add_scalar('Physics/nbody_raw', nbody_loss_raw, epoch_idx)
            writer.add_scalar('Physics/nbody_weight', nbody_weight_eff, epoch_idx)
            writer.add_scalar('Physics/nbody_weighted', nbody_loss_weighted, epoch_idx)
            
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
    
    if rank == 0 and writer is not None:
        writer.close()
```

### Launch TensorBoard

```bash
# Start TensorBoard server
tensorboard --logdir=runs --port=6006 --bind_all

# Access at: http://localhost:6006
# Or for remote server: ssh -L 6006:localhost:6006 user@server
```

### TensorBoard Features

1. **Scalars**: Training curves, metrics over time
2. **Graphs**: Model architecture visualization
3. **Distributions**: Weight/gradient distributions
4. **Histograms**: Parameter evolution
5. **Profiler**: Performance bottleneck analysis
6. **HParams**: Hyperparameter comparison

---

## Weights & Biases Setup

### Installation

```bash
pip install wandb
wandb login  # Follow prompts to authenticate
```

### Implementation

```python
import wandb

def train_emulator(..., use_wandb=True, wandb_project="pinn-solar-system"):
    """Train with W&B logging."""
    
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
            'num_bodies': len(bodies),
            'num_samples': len(dataset['times_seconds']),
        }
        
        run = wandb.init(
            project=wandb_project,
            config=config,
            name=f"ddp_gpu{world_size}_{datetime.now().strftime('%m%d_%H%M')}",
            tags=['ddp', f'{world_size}gpu', cfg.training_mode],
        )
        
        # Watch model (logs gradients and parameters)
        wandb.watch(model, log='all', log_freq=100)
    
    for epoch_idx in epoch_iter:
        # ... training loop ...
        
        if rank == 0 and use_wandb:
            # Log metrics
            metrics = {
                'epoch': epoch_idx,
                'train/loss': epoch_train_loss,
                'train/objective_loss': epoch_objective_loss,
                'val/loss': epoch_val_loss,
                'val/pos_rmse_km': val_pos_rmse,
                'val/vel_rmse_km_s': val_vel_rmse,
                'physics/nbody_loss': nbody_loss_raw,
                'physics/nbody_weight': nbody_weight_eff,
                'physics/energy_loss': energy_loss,
                'physics/angular_momentum_loss': angular_momentum_loss,
                'optim/lr': optimizer.param_groups[0]['lr'],
                'optim/grad_norm': total_grad_norm,
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
    
    if rank == 0 and use_wandb:
        # Save final checkpoint to W&B
        wandb.save(str(checkpoint_path))
        
        # Log summary metrics
        wandb.run.summary['best_val_loss'] = best_val
        wandb.run.summary['total_epochs'] = epoch_idx + 1
        
        wandb.finish()
```

### W&B Features

1. **Real-time Metrics**: Live updating charts
2. **Hyperparameter Sweeps**: Automated HPO
3. **Artifact Tracking**: Version control for datasets/models
4. **Reports**: Share results with team
5. **Alerts**: Get notified of failures/milestones

---

## Per-Rank Logging

### Implementation

Create `src/solsys_emulator/distributed_logger.py`:

```python
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


# Usage in train.py:
def train_emulator(...):
    # Create logger
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
    
    for epoch_idx in epoch_iter:
        logger.info(f"Epoch {epoch_idx + 1}/{cfg.epochs}")
        
        for batch_idx, (batch_t, batch_y) in enumerate(train_loader):
            # Log batch stats (debug)
            if batch_idx % 100 == 0:
                logger.log_batch_stats(batch_t, batch_idx)
            
            # ... training step ...
            
            # Log gradients
            if batch_idx % 100 == 0:
                logger.log_gradients(model, batch_idx)
            
            # Log GPU memory
            if batch_idx % 100 == 0:
                logger.log_gpu_memory(batch_idx)
        
        logger.info(
            f"Epoch {epoch_idx + 1} complete | "
            f"train_loss={epoch_train_loss:.6f}, "
            f"val_loss={epoch_val_loss:.6f}"
        )
    
    logger.sync_and_log("Training complete!")
```

---

## GPU Memory Profiling

### PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

def train_emulator(..., enable_profiling=False):
    """Train with profiling enabled."""
    
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
    
    with profiler_ctx if profiler_ctx else nullcontext():
        for epoch_idx in range(cfg.epochs):
            for batch_idx, (batch_t, batch_y) in enumerate(train_loader):
                # Training step
                with record_function("forward"):
                    pred = model(batch_t)
                    loss = criterion(pred, batch_y)
                
                with record_function("backward"):
                    loss.backward()
                
                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()
                
                if profiler_ctx:
                    profiler_ctx.step()
```

View profiling results:
```bash
tensorboard --logdir=./profiler_logs
# Navigate to "PyTorch Profiler" tab
```

### Memory Snapshots

```python
import torch.cuda

# Take memory snapshot
if rank == 0 and epoch_idx % 100 == 0:
    torch.cuda.memory._dump_snapshot(f"memory_snapshot_epoch{epoch_idx}.pickle")

# Analyze snapshot (in notebook or script):
import pickle
with open("memory_snapshot_epoch100.pickle", "rb") as f:
    snapshot = pickle.load(f)

# Use pytorch.org/memory_viz to visualize
```

---

## Debugging Workflows

### Debugging Checklist

1. **Verify Single-GPU Baseline**
   ```bash
   python train.py --device cuda --epochs 10 --batch-size 128
   ```

2. **Test DDP with 1 GPU** (should match baseline)
   ```bash
   torchrun --nproc_per_node=1 train_ddp.py --epochs 10 --batch-size 128
   ```

3. **Test DDP with 2 GPUs**
   ```bash
   torchrun --nproc_per_node=2 train_ddp.py --epochs 10 --batch-size 128
   ```

4. **Check logs** in `logs/rank_*.log`

5. **Compare metrics** (loss, RMSE should be within 1-2%)

### Common Debug Scenarios

#### Scenario 1: Loss Diverges in DDP

**Check**:
```bash
# Compare per-rank losses
grep "train_loss" logs/rank_0.log > rank0_loss.txt
grep "train_loss" logs/rank_1.log > rank1_loss.txt
diff rank0_loss.txt rank1_loss.txt
```

**Fix**: Ensure DistributedSampler is used and `drop_last=True`

#### Scenario 2: Hangs/Deadlocks

**Debug**:
```python
# Add to train loop
import signal
import sys

def timeout_handler(signum, frame):
    print(f"[Rank {rank}] TIMEOUT at line {frame.f_lineno}")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute timeout

# Training code here...

signal.alarm(0)  # Disable alarm
```

**Fix**: Add `dist.barrier()` to synchronize ranks

#### Scenario 3: OOM on One GPU

**Debug**:
```python
# Log memory after each major operation
def log_memory(msg):
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"[Rank {rank}] {msg}: {allocated:.2f}GB")

log_memory("Before forward")
pred = model(batch_t)
log_memory("After forward")
loss = criterion(pred, batch_y)
log_memory("After loss")
loss.backward()
log_memory("After backward")
```

**Fix**: Reduce batch size or enable gradient checkpointing

---

## Production Monitoring

### System Monitor Script

Create `monitor_training.py`:

```python
#!/usr/bin/env python3
"""Monitor multi-GPU training in real-time."""

import subprocess
import time
from pathlib import Path


def get_gpu_stats():
    """Get GPU utilization and memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().split('\n')
    except Exception as e:
        return [f"Error: {e}"]


def tail_logs(log_dir: Path, n_lines: int = 5):
    """Tail the last n lines from each rank log."""
    logs = {}
    for log_file in sorted(log_dir.glob("rank_*.log")):
        rank = log_file.stem.split('_')[1]
        try:
            with open(log_file) as f:
                logs[rank] = f.readlines()[-n_lines:]
        except Exception as e:
            logs[rank] = [f"Error: {e}"]
    return logs


def main():
    log_dir = Path("logs")
    
    print("=" * 80)
    print("MULTI-GPU TRAINING MONITOR")
    print("=" * 80)
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            # Clear screen
            print("\033[2J\033[H", end="")
            
            # GPU stats
            print("=" * 80)
            print("GPU STATUS")
            print("=" * 80)
            print("GPU | Util% | Memory")
            print("-" * 40)
            for line in get_gpu_stats():
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu_id, util, mem_used, mem_total = parts[:4]
                        print(f"{gpu_id:3s} | {util:5s} | {mem_used:7s}/{mem_total:7s} MB")
            
            # Log tails
            print("\n" + "=" * 80)
            print("RECENT LOGS")
            print("=" * 80)
            logs = tail_logs(log_dir, n_lines=3)
            for rank, lines in logs.items():
                print(f"\nRank {rank}:")
                for line in lines:
                    print(f"  {line.rstrip()}")
            
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
```

Run with:
```bash
python monitor_training.py
```

### Slack/Email Alerts

```python
import requests

def send_slack_alert(webhook_url: str, message: str):
    """Send alert to Slack channel."""
    payload = {"text": message}
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Slack alert: {e}")

# Usage in train.py:
if rank == 0:
    try:
        # Training code...
        pass
    except Exception as e:
        send_slack_alert(
            webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            message=f"🚨 Training failed on {os.getenv('HOSTNAME')}: {e}"
        )
        raise
    
    # Success notification
    send_slack_alert(
        webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        message=f"✅ Training complete! Best val loss: {best_val:.6f}"
    )
```

---

## Summary

### Recommended Setup

For production multi-GPU training:

1. **TensorBoard** for visualizations (always)
2. **W&B** for experiment tracking (recommended)
3. **Per-rank logs** for debugging (essential)
4. **Profiler** for performance optimization (as needed)
5. **Monitoring script** for long runs (helpful)

### Launch Example with Full Logging

```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --training-mode unified \
    --batch-size 768 \
    --epochs 1400 \
    --log-dir logs/run_$(date +%Y%m%d_%H%M%S) \
    --wandb-project pinn-solar-system \
    --wandb-entity your-team

# In another terminal:
tensorboard --logdir runs --port 6006
python monitor_training.py
```

### Quick Reference

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **Per-rank logs** | Debug individual GPUs | Always |
| **TensorBoard** | Visualize training | Always |
| **W&B** | Track experiments | Recommended |
| **Profiler** | Find bottlenecks | When optimizing |
| **Monitor script** | Watch live progress | Long training runs |
| **Alerts** | Get notified | Production |
