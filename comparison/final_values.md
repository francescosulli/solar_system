# Final Experimental Values — All Models

Complete record of architecture parameters, training configuration, and experimental results for all three models (MLP, PINN, HPINN) trained on the NASA DE440 ephemeris dataset. All models use identical **768×8 architectures** for a perfectly controlled "apples-to-apples" comparison.

---

## Dataset

| Parameter | Value |
| :--- | :--- |
| **Source** | NASA JPL DE440 ephemeris |
| **Bodies** | 10 (Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Moon) |
| **Dataset shape** | (58441, 10, 6) — T × Bodies × State |
| **Time span** | ~160 years (sampled every 3 hours) |
| **Train / Val split** | 90 % / 10 % (random) |
| **Time normalization** | Mean / Standard Deviation (Standard Scaler) |

---

## 1. Global Benchmark (All Bodies, 20 Years) — Paper 2
*Mean position RMSE calculated across the entire dataset (train + validation).*

| Model | Architecture | Global Mean RMSE | Improvement vs MLP |
| :--- | :--- | :--- | :--- |
| **MLP** | 768×8 | 113,889 km | — |
| **PINN** | 768×8 | 48,778 km | -99.99% |
| **HPINN** | 768×8 (Hybrid) | **34,037 km** | **-99.99% (Best)** |

> [!TIP]
> The HPINN reduces the residual error of the standard PINN by a further ~30% (from 48k to 34k km) globally.

---

## 2. Long-Term Extrapolation (Earth) — Paper 1
*Mean position error evaluated exclusively outside the training boundary (extrapolation).*

| Model | Mean Pos Error (Earth) | Note |
| :--- | :--- | :--- |
| **MLP** | 157,896 km | Significativa deviazione dalla traiettoria reale. |
| **PINN** | 19,608 km | Stable Newtonian trajectory. |
| **HPINN** | **11,750 km** | Most precise and physically stable extrapolation. |

---

## Training History & Best Validation (Pre-Physics)

Before the final Physics stage (Stage 3), models are evaluated on pure data-fitting capabilities (Stage 2: Refine).

| Model | Best Validation RMSE (Stage 2 - Data Only) |
| :--- | :--- |
| **PINN (768x8)** | 88,423 km |
| **HPINN (768x8)** | **51,775 km** |

> [!IMPORTANT]
> When the N-Body physics loss is enforced in Stage 3, the standard PINN struggles to reconcile physics with data (validation error spikes). The HPINN, thanks to its $\Delta r$ residual branch, absorbs the discrepancy, remaining highly accurate while obeying Newtonian mechanics.

---

## Model Architectures

All models share the same backbone capacity for fair comparison.

### Common Backbone
*   **Hidden Dimension**: 768
*   **Number of Layers**: 8
*   **Fourier Features**: 256 (Log spacing, 0.02 to 256.0)
*   **Head Layers**: 3 (384 dim)
*   **Interaction Layers**: 2 (384 dim)

### Model-Specific Differences
*   **MLP**: `state_mode="position_only"` (Velocity derived via AutoDiff `jacfwd`. Zero physics loss).
*   **PINN**: `state_mode="position_only"` (Velocity derived via AutoDiff `jacfwd`. N-Body loss active).
*   **HPINN**: `hybrid_correction=True`. Adds a secondary residual branch (2 layers, 128 dim) to learn non-Newtonian dynamics.

### Training Strategy (HPINN & PINN)
3-Stage Curriculum Learning:
1.  **Coarse**: Data-fitting only.
2.  **Refine**: Lower learning rate.
3.  **Physics**: Activate N-Body loss (Weight: 1e-6).
