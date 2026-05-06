# PINN Training - Quick Reference Card

## Single-GPU Training

```bash
# Basic training (auto-detects GPU)
python train.py --training-mode unified --device auto

# Specify GPU
python train.py --training-mode unified --device cuda

# Custom configuration
python train.py \
    --training-mode unified \
    --device cuda \
    --epochs 1400 \
    --batch-size 1536 \
    --collocation-points 512

# CPU training (for testing)
python train.py --training-mode unified --device cpu --epochs 10
```

## Multi-Stage Training

```bash
# Full three-stage training
python train.py --training-mode multi --device cuda

# Skip physics stage (faster, but less accurate)
python train.py --training-mode multi --device cuda --skip-physics-stage

# Resume interrupted training (automatic)
python train.py --training-mode multi --device cuda
```

## Multi-GPU Training (After DDP Implementation)

```bash
# 2 GPUs on single node
torchrun --nproc_per_node=2 train_ddp.py \
    --training-mode unified \
    --batch-size 768 \
    --epochs 1400

# 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py \
    --training-mode unified \
    --batch-size 384

# With logging
torchrun --nproc_per_node=2 train_ddp.py \
    --training-mode unified \
    --batch-size 768 \
    --log-dir logs/run_$(date +%Y%m%d_%H%M%S) \
    --wandb-project pinn-solar-system
```

## Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor training logs (in separate terminal)
python monitor_training.py

# TensorBoard (after enabling in code)
tensorboard --logdir runs --port 6006

# SSH tunnel for remote TensorBoard
ssh -L 6006:localhost:6006 user@server
# Then open http://localhost:6006 in browser
```

## Troubleshooting

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Check number of GPUs
python -c "import torch; print(torch.cuda.device_count())"

# Check per-rank logs (DDP mode)
tail -f logs/rank_0.log
tail -f logs/rank_1.log

# Compare losses across ranks
grep "train_loss" logs/rank_*.log | head -20

# Test DDP with minimal example
torchrun --nproc_per_node=2 minimal_ddp_example.py
```

## Common Issues and Fixes

### "CUDA out of memory"
```bash
# Solution 1: Reduce batch size
python train.py --training-mode unified --batch-size 768

# Solution 2: Use multi-stage training
python train.py --training-mode multi

# Solution 3: Use 2 GPUs (after DDP implementation)
torchrun --nproc_per_node=2 train_ddp.py --batch-size 768
```

### "Training is slow"
```bash
# Check GPU utilization (should be >80%)
nvidia-smi

# Increase DataLoader workers
python train.py --training-mode unified --loader-workers 8

# Check if using GPU
python train.py --device cuda  # explicitly set
```

### "DDP hangs"
```bash
# Check all ranks started
ps aux | grep train_ddp.py

# Look for errors in per-rank logs
tail -n 50 logs/rank_*.log

# Test with minimal example first
torchrun --nproc_per_node=2 minimal_ddp_example.py
```

## File Locations

```
artifacts/
├── checkpoint.pt                    # Final checkpoint
├── checkpoint_coarse.pt             # Multi-stage: coarse
├── checkpoint_refine.pt             # Multi-stage: refine
├── checkpoint_physics.pt            # Multi-stage: physics
├── history_coarse.json              # Multi-stage: coarse history
├── history_refine.json              # Multi-stage: refine history
├── history_physics.json             # Multi-stage: physics history
├── training_history.png             # Training curves
├── validation_rmse.png              # Validation metrics
├── nbody_diagnostics.png            # Physics diagnostics
└── training_summary.json            # Run metadata

logs/
├── rank_0.log                       # DDP: Rank 0 detailed log
├── rank_1.log                       # DDP: Rank 1 detailed log
└── ...

runs/
└── [timestamp]_gpu[N]/              # TensorBoard logs
    ├── events.out.tfevents.*
    └── ...
```

## Configuration Examples

### Maximum Performance (2 GPUs)
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --training-mode unified \
    --batch-size 1536 \
    --epochs 1400 \
    --loader-workers 8 \
    --collocation-points 512
```

### Memory-Constrained (Single GPU)
```bash
python train.py \
    --training-mode multi \
    --device cuda \
    --batch-size 768
```

### Quick Test Run
```bash
python train.py \
    --training-mode unified \
    --device cuda \
    --epochs 10 \
    --batch-size 128
```

### Full Production Run with Logging
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --training-mode unified \
    --batch-size 768 \
    --epochs 1400 \
    --log-dir logs/production_run \
    --wandb-project pinn-solar-system \
    --wandb-entity your-team
```

## Performance Expectations

| Setup | Batch/GPU | Total Batch | Time/Epoch* | Memory/GPU |
|-------|-----------|-------------|-------------|------------|
| 1 GPU | 1536 | 1536 | 100s | ~40GB (OOM!) |
| 1 GPU Multi-Stage | 1536 | 1536 | 90s | ~35GB |
| 2 GPU DDP | 768 | 1536 | 55s | ~24GB ✓ |
| 2 GPU DDP | 1536 | 3072 | 90s | ~40GB |
| 4 GPU DDP | 768 | 3072 | 30s | ~24GB ✓ |

*Approximate times for A100 40GB GPUs

## Documentation

- **README.md** - Project overview and quick start
- **DDP_IMPLEMENTATION.md** - Complete DDP guide (3000+ words)
- **DDP_PATCH.md** - Specific code changes needed
- **LOGGING_GUIDE.md** - Multi-GPU logging setup
- **QUICK_REFERENCE.md** - This file

## Getting Help

1. Check the appropriate guide:
   - Memory issues → DDP_IMPLEMENTATION.md (Memory Optimization)
   - Multi-GPU setup → DDP_PATCH.md
   - Debugging → LOGGING_GUIDE.md (Debugging Workflows)

2. Run comparison script:
   ```bash
   python compare_approaches.py
   ```

3. Test with minimal example:
   ```bash
   torchrun --nproc_per_node=2 minimal_ddp_example.py
   ```

4. Check logs:
   ```bash
   tail -f logs/rank_0.log  # Main process
   tail -f logs/rank_1.log  # Worker process
   ```
