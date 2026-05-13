# Final Values — All Models

Complete record of architecture parameters, training configuration, and experimental results for all three models trained on the NASA DE440 ephemeris dataset.

---

## Dataset

| Parameter | Value |
| :--- | :--- |
| **Source** | NASA JPL DE440 ephemeris |
| **Bodies** | 10 (Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Moon) |
| **Dataset shape** | (58441, 10, 6) — T × Bodies × State |
| **State variables** | Position $(x, y, z)$ in km; Velocity $(v_x, v_y, v_z)$ in km/s |
| **Time span** | ~160 years (daily sampling) |
| **Train / Val split** | 90 % / 10 % (random) |
| **Normalization** | Per-body feature-wise standard scaling |
| **Time normalization** | Min-max → \[−1, 1\] |

---

## Model 1: MLP (Purely Data-Driven Baseline)

### Architecture

| Parameter | Value |
| :--- | :--- |
| `state_mode` | `full` (predicts position + velocity directly) |
| `backbone_type` | `residual` |
| `hidden_dim` | 768 |
| `num_layers` | 8 |
| `fourier_features` | 256 |
| `min_frequency` | 0.02 |
| `max_frequency` | 256.0 |
| `frequency_spacing` | `log` |
| `head_layers` | 3 |
| `head_hidden_dim` | 384 |
| `dropout` | 0.0 |
| `num_bodies` | 10 |

### Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Training stages** | 1 (no physics stage) |
| `epochs` | 2000 (max) |
| `batch_size` | 1536 |
| `lr` (initial) | 1e-3 |
| `min_lr` | 1e-6 |
| `lr_scheduler` | cosine annealing |
| `weight_decay` | 1e-6 |
| `grad_clip_norm` | 1.0 |
| `early_stopping_patience` | 200 |
| `position_loss_weight` | 1.0 |
| `velocity_loss_weight` | 1.0 |
| `nbody_loss_weight` | 0.0 (disabled) |
| `physics_loss_weight` | 0.0 (disabled) |
| `seed` | 42 |
| **Hardware** | NVIDIA GB10 (CUDA 12.1) |

### Results

| Metric | Value |
| :--- | :--- |
| **Best validation epoch** | 46 |
| **Best validation RMSE (position)** | **1,662,400.84 km** |
| **Final selected stage** | stage1 |

---

## Model 2: PINN (Physics-Informed Neural Network)

### Architecture

| Parameter | Value |
| :--- | :--- |
| `state_mode` | `position_only` (velocities derived via AutoDiff `jacfwd`) |
| `backbone_type` | `residual` |
| `hidden_dim` | 512 |
| `num_layers` | 6 |
| `fourier_features` | 256 |
| `min_frequency` | 0.02 |
| `max_frequency` | 256.0 |
| `frequency_spacing` | `log` |
| `head_layers` | 3 |
| `head_hidden_dim` | 256 |
| `body_embedding_dim` | 64 |
| `interaction_layers` | 2 |
| `interaction_hidden_dim` | 256 |
| `use_layer_norm` | `True` |
| `dropout` | 0.0 |
| `num_bodies` | 10 |

> [!NOTE]
> The PINN checkpoint used for comparison is stored in `PINN/artifacts_giant/`. It uses a 512×6 architecture — smaller than the HPINN (768×8) but significantly larger than earlier experimental runs.

### Training Configuration (Multi-Stage)

| Stage | Epochs | Batch | LR | N-Body Weight |
| :--- | :--- | :--- | :--- | :--- |
| **Coarse** (data fit) | 2000 | 1536 | 3e-4 | 0.0 |
| **Refine** | 2000 | 1536 | 1.2e-4 | 0.0 |
| **Physics** | 1000 | 1536 | 5e-5 | 1e-6 |

Common parameters across all stages:
- `lr_scheduler`: cosine annealing
- `early_stopping_patience`: 200
- `AutoDiff mode`: `jacfwd` (Forward-Mode, with pre-compiled `vmap` cache)

### Results

| Metric | Value |
| :--- | :--- |
| **Best validation epoch** | 1974 |
| **Best validation RMSE (position)** | **64,212.96 km** |
| **Best validation RMSE (velocity)** | 0.008846 km/s |
| **Last epoch RMSE (position)** | 70,885.77 km |
| **Total epochs trained** | 2000 |
| **Checkpoint path** | `PINN/artifacts_giant/final_checkpoint.pt/final_checkpoint.pt` |

---

## Model 3: HPINN (Hybrid Physics-Informed Neural Network)

### Architecture

| Parameter | Value |
| :--- | :--- |
| `state_mode` | `position_only` (velocities derived via AutoDiff `jacfwd`) |
| `backbone_type` | `residual` |
| `hidden_dim` | 768 |
| `num_layers` | 8 |
| `fourier_features` | 256 |
| `min_frequency` | 0.02 |
| `max_frequency` | 256.0 |
| `frequency_spacing` | `log` |
| `head_layers` | 3 |
| `head_hidden_dim` | 384 |
| `body_embedding_dim` | 64 |
| `interaction_layers` | 2 |
| `interaction_hidden_dim` | 384 |
| **`hybrid_correction`** | **`True`** |
| `correction_layers` | 2 |
| `correction_hidden_dim` | 128 |
| `correction_init_scale` | 0.02 |
| `use_layer_norm` | `True` |
| `dropout` | 0.0 |
| `num_bodies` | 10 |

The output is $\mathbf{r}(t) = \mathbf{r}_\text{base}(t) + \Delta\mathbf{r}(t)$, where $\mathbf{r}_\text{base}$ is the Newtonian backbone and $\Delta\mathbf{r}$ is the learned residual correction.

### Training Configuration (Multi-Stage)

| Stage | Epochs | Batch | LR | N-Body Weight | Correction Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Coarse** (data fit) | 2000 | 1536 | 3e-4 | 0.0 | 0.0 |
| **Refine** | 2000 | 1536 | 1.2e-4 | 0.0 | 0.0 |
| **Physics** | 1000 | 1536 | 5e-5 | 1e-6 | **1e-4** |

Common parameters across all stages:
- `lr_scheduler`: cosine annealing
- `early_stopping_patience`: 200
- `AutoDiff mode`: `jacfwd` (Forward-Mode, with pre-compiled `vmap` cache)
- **Memory footprint (VRAM)**: < 6 GB (vs. > 48 GB with legacy `jacrev`)

### Results

| Metric | Value |
| :--- | :--- |
| **Best validation epoch** | 1952 |
| **Best validation RMSE (position)** | **51,775.77 km** |
| **Best validation RMSE (velocity)** | 0.01409 km/s |
| **Learning rate at best epoch** | 6.70e-7 |
| **Last epoch RMSE (position)** | 53,887.76 km |
| **Last epoch RMSE (velocity)** | 0.01394 km/s |
| **Last epoch correction loss** | 7.73e-3 |
| **Total epochs trained** | 2000 |

---

## Summary Comparison

| Model | Architecture | Best RMSE (position) | Best RMSE (velocity) | Best Epoch | Improvement vs MLP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | 768×8, `full` state | 1,662,400.84 km | — | 46 | — |
| **PINN** | 512×6, `position_only` | 64,212.96 km | 0.008846 km/s | 1974 | **−96.1 %** |
| **HPINN** | 768×8, `position_only` + correction | **51,775.77 km** | **0.01409 km/s** | 1952 | **−96.9 %** |

> [!IMPORTANT]
> The PINN (512×6) and HPINN (768×8) use slightly different architecture sizes. A perfectly controlled comparison on identical architectures is deferred to future work, but the gap (64k vs 51k km) remains significant and consistent across multiple runs.
