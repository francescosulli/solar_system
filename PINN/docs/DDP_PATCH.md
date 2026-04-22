# DDP Modifications for src/solsys_emulator/train.py
# Apply these changes to enable multi-GPU training

## STEP 1: Add imports at the top of the file (after line 13)

```python
# Add these imports after the existing torch imports
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

## STEP 2: Add DDP fields to TrainConfig (after line 84)

```python
@dataclass
class TrainConfig:
    """Training hyper-parameters."""
    
    # ... existing fields ...
    show_progress: bool = True
    
    # === ADD THESE DDP FIELDS ===
    distributed: bool = False  # Enable DDP mode
    local_rank: int = 0  # Local GPU rank (set by torchrun)
    find_unused_parameters: bool = False  # For models with conditional paths
```

## STEP 3: Modify train_emulator function signature (around line 420)

Find the function definition and add rank/world_size tracking:

```python
def train_emulator(
    dataset: dict[str, Any],
    train_config: TrainConfig,
    model_config: ModelConfig,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    initial_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Train the ephemeris emulator with optional DDP support."""
    
    cfg = train_config
    mcfg = model_config
    
    # === ADD DDP INITIALIZATION HERE ===
    rank = 0
    world_size = 1
    is_distributed = cfg.distributed
    
    if is_distributed:
        if not dist.is_initialized():
            raise RuntimeError(
                "DDP mode enabled but torch.distributed not initialized. "
                "Launch with: torchrun --nproc_per_node=N train_ddp.py"
            )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = cfg.local_rank
        
        # Set device to local rank
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
        # Helper for rank-0 only printing
        def print_rank0(*args, **kwargs):
            if rank == 0:
                print(*args, **kwargs)
    else:
        device = torch.device(cfg.device)
        print_rank0 = print
    
    print_rank0(f"[Rank {rank}/{world_size}] Training on {device}")
    
    # ... rest of existing setup code ...
    # (keep all the existing dataset processing, normalization, etc.)
```

## STEP 4: Modify DataLoader creation (around line 530)

Replace the DataLoader creation with DDP-aware version:

```python
# === REPLACE THIS SECTION ===
# OLD CODE (around line 530-548):
#   train_loader = DataLoader(TensorDataset(...), batch_size=..., shuffle=...)
#   val_loader = DataLoader(TensorDataset(...), batch_size=..., shuffle=False)

# NEW CODE WITH DDP SUPPORT:
    # Create datasets
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
```

## STEP 5: Wrap model with DDP (around line 551)

After model creation and before optimizer creation:

```python
    # Create model
    model = EmulatorModel(**mcfg.to_kwargs()).to(device)
    
    # Load initial checkpoint if provided
    if initial_checkpoint_path is not None:
        init_payload = load_checkpoint(initial_checkpoint_path, map_location=str(device))
        # ... existing checkpoint loading code ...
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
    
    # ... continue with optimizer creation ...
    # NOTE: optimizer uses model.parameters() which works with both DDP and non-DDP
```

## STEP 6: Update training loop to set epoch for sampler (around line 621)

At the start of the epoch loop:

```python
    for epoch_idx in epoch_iter:
        # === ADD THIS ===
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        
        # ... existing warmup/weight calculations ...
```

## STEP 7: Aggregate training metrics (around line 800)

After the training loop accumulates metrics, aggregate across GPUs:

```python
        # Training loop
        for batch_t, batch_y in train_loader:
            # ... training code accumulates running_train, running_objective, etc. ...
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
        
        # Calculate epoch metrics (n_batches now reflects all ranks)
        epoch_train_loss = running_train / max(1, n_batches)
        # ... other epoch metrics ...
```

## STEP 8: Aggregate validation metrics (around line 900)

Similar aggregation for validation:

```python
        # Validation loop
        model.eval()
        val_running = 0.0
        val_pos_error_sum = 0.0
        val_vel_error_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch_t, batch_y in val_loader:
                # ... validation code ...
                val_count += batch_t.shape[0]
        
        # === ADD VALIDATION METRIC AGGREGATION ===
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
        
        # Calculate validation metrics
        epoch_val_loss = val_running / max(1, len(val_loader) * world_size)
        # ... other validation metrics ...
```

## STEP 9: Save checkpoint from rank 0 only (around line 1000)

Modify checkpoint saving:

```python
        # Track best model
        if epoch_val_metric < best_val:
            best_val = epoch_val_metric
            
            # === MODIFIED: Get state dict correctly for DDP ===
            if is_distributed:
                # DDP wraps the model, access .module to get underlying model
                best_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
            else:
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
            stale_epochs = 0
            print_rank0(f"  → New best {cfg.selection_metric}: {best_val:.6f}")
        # ... early stopping logic ...
    
    # Final checkpoint save
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
    
    return {"history": history, "scaler": scaler_obj}
```

## Summary of Changes

**Total modifications**: ~100 lines added/changed across 9 steps
**Backward compatible**: Works with or without DDP (controlled by `distributed` flag)
**Key additions**:
1. DDP imports and config fields
2. Rank/world_size tracking
3. DistributedSampler for data loading
4. DDP model wrapping
5. Metric aggregation across GPUs
6. Rank-0 only checkpoint saving

**Testing checklist**:
1. Non-DDP mode: `python train.py` (should work exactly as before)
2. DDP mode: `torchrun --nproc_per_node=1 train_ddp.py` (should match non-DDP)
3. Multi-GPU: `torchrun --nproc_per_node=2 train_ddp.py` (should split work)
