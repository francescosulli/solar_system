#!/usr/bin/env python3
"""
Paper 1: Long-Term Stability in Data-Driven and Physics-Informed Solar-System Neural Models
Generates 3 publication-ready PDF plots:
  1. Extrapolation Error vs. Time
  2. Orbital Energy Drift (Delta E / E0)
  3. Angular Momentum Conservation
"""
import gc
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
COMP_DIR    = SCRIPT_DIR.parent
WORKSPACE   = COMP_DIR.parent
PLOTS_DIR   = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.append(str(WORKSPACE / "HPINN" / "src"))
from solsys_emulator.de440_dataset import load_dataset
from solsys_emulator.model import EmulatorModel, ModelConfig

DATASET_PATH = WORKSPACE / "data" / "dataset_de440.npz"

CHECKPOINTS = {
    "MLP":   WORKSPACE / "MLP"  / "artifacts" / "emulator_stage1.pt",
    "PINN":  WORKSPACE / "PINN" / "artifacts_768x8" / "final_checkpoint.pt" / "checkpoint_physics.pt",
    "HPINN": WORKSPACE / "HPINN"/ "artifacts" / "final_checkpoint.pt",
}

MODEL_COLORS = {
    "MLP":   "#d62728",   # red
    "PINN":  "#1f77b4",   # blue
    "HPINN": "#2ca02c",   # green
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_FRAC = 0.90   # first 90 % = training window

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.size":        12,
        "axes.titlesize":   14,
        "axes.labelsize":   12,
        "legend.fontsize":  11,
        "figure.dpi":       300,
    })

# ---------------------------------------------------------------------------
# Model loading (handles legacy key differences between MLP / PINN / HPINN)
# ---------------------------------------------------------------------------
def load_model(name: str, path: Path):
    if not path.exists():
        print(f"  [SKIP] {name}: checkpoint not found at {path}")
        return None, None, None, None

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # --- model config ---
    cfg_dict = dict(ckpt.get("model_kwargs", ckpt.get("model_config", {})))
    if name in ("PINN", "HPINN"):
        cfg_dict["state_mode"] = "position_only"
    if name == "HPINN":
        cfg_dict["hybrid_correction"] = True
    if name == "MLP":
        cfg_dict["backbone_type"] = "residual"

    cfg = ModelConfig(**cfg_dict)
    model = EmulatorModel(**cfg.to_kwargs()).to(DEVICE)

    # --- state dict (fix legacy key naming) ---
    state_dict = dict(ckpt.get("model_state_dict", ckpt.get("model_state", {})))
    if name == "MLP":
        remapped = {}
        for k, v in state_dict.items():
            new_k = k.replace(".linear1.", ".fc1.").replace(".linear2.", ".fc2.")
            if "backbone." in new_k:
                parts = new_k.split(".")
                try:
                    idx = int(parts[1])
                    if idx >= 2:
                        parts[1] = str(idx - 1)
                        new_k = ".".join(parts)
                except ValueError:
                    pass
            remapped[new_k] = v
        state_dict = remapped
    model.load_state_dict(state_dict)
    model.eval()

    # --- scaler ---
    if "scaler_mean" in ckpt:
        s_mean = torch.from_numpy(ckpt["scaler_mean"]).float().to(DEVICE)
        s_std  = torch.from_numpy(ckpt["scaler_std"]).float().to(DEVICE)
    else:
        s_mean = torch.tensor(ckpt["scaler"]["mean"]).float().to(DEVICE)
        s_std  = torch.tensor(ckpt["scaler"]["std"]).float().to(DEVICE)

    return model, cfg, s_mean, s_std

# ---------------------------------------------------------------------------
# Inference  →  predicted states [T, B, 6] in km / km·s⁻¹
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_states(name, model, cfg, s_mean, s_std, t_norm_t, time_std):
    chunk = 500
    all_pos, all_vel = [], []

    if cfg.state_mode == "position_only":
        from torch import vmap
        from torch.func import jacfwd
        def _single(t1d):
            return model(t1d.unsqueeze(0)).squeeze(0)
        _vmap_vel = vmap(jacfwd(_single))

    for i in range(int(np.ceil(len(t_norm_t) / chunk))):
        t_ch = t_norm_t[i*chunk:(i+1)*chunk].to(DEVICE)

        if cfg.state_mode == "full":
            out = model(t_ch) * s_std + s_mean
            all_pos.append(out[..., :3].cpu())
            all_vel.append(out[..., 3:].cpu())
        else:
            if name == "HPINN":
                pos_norm, _, _ = model.forward_components(t_ch)
            else:
                pos_norm = model(t_ch)

            vel_norm = _vmap_vel(t_ch)
            pos = pos_norm * s_std[..., :3] + s_mean[..., :3]
            vel = vel_norm * (s_std[..., :3] / time_std)
            all_pos.append(pos.cpu())
            all_vel.append(vel.cpu())

    pos_km  = torch.cat(all_pos, 0).numpy()
    vel_kms = torch.cat(all_vel, 0).numpy()
    return np.concatenate([pos_km, vel_kms], axis=-1)   # [T, B, 6]

# ---------------------------------------------------------------------------
# Physical diagnostics
# ---------------------------------------------------------------------------
MU_SUN = 132_712_440_041.9394   # km³ s⁻²

def specific_energy(states):
    """Specific orbital energy [T, B]."""
    pos = states[..., :3]
    vel = states[..., 3:]
    r   = np.linalg.norm(pos, axis=-1)
    v2  = np.sum(vel**2, axis=-1)
    return 0.5 * v2 - MU_SUN / r

def angular_momentum_magnitude(states):
    """Magnitude of specific angular momentum L = r × v [T, B]."""
    pos = states[..., :3]   # [T, B, 3]
    vel = states[..., 3:]
    L   = np.cross(pos, vel)  # [T, B, 3]
    return np.linalg.norm(L, axis=-1)  # [T, B]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_style()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print("Loading dataset…")
    dataset   = load_dataset(DATASET_PATH)
    times_sec = dataset["times_seconds"]              # [T]
    true_st   = dataset["states"]                     # [T, B, 6]
    bodies    = list(dataset["bodies"])
    
    BODY_IDX = 3   # Earth (index in DE440 default order)
    BODY_NAME = bodies[BODY_IDX] if BODY_IDX < len(bodies) else "body_3"

    t_mean = float(np.mean(times_sec))
    t_std  = float(np.std(times_sec))
    t_norm = (times_sec - t_mean) / t_std
    t_norm_t = torch.from_numpy(t_norm).float()

    t_years   = times_sec / (365.25 * 86400)
    split_idx = int(len(times_sec) * TRAIN_FRAC)

    # --- true diagnostics ---
    true_E = specific_energy(true_st)       # [T, B]
    true_L = angular_momentum_magnitude(true_st)  # [T, B]

    predictions = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"Loading & running inference: {name}…")
        model, cfg, s_mean, s_std = load_model(name, ckpt_path)
        if model is None:
            continue
        predictions[name] = predict_states(name, model, cfg, s_mean, s_std, t_norm_t, t_std)
        
        # --- Print stats to verify distinct models ---
        err_pos = np.linalg.norm(predictions[name][:, BODY_IDX, :3] - true_st[:, BODY_IDX, :3], axis=-1)
        print(f"  [{name}] Stats:")
        print(f"    Mean Pos Error: {np.mean(err_pos):.2f} km")
        print(f"    Max  Pos Error: {np.max(err_pos):.2f} km")
        print(f"    Config: {cfg.hidden_dim}x{cfg.num_layers}, hybrid={getattr(cfg, 'hybrid_correction', False)}")
        
        del model; torch.cuda.empty_cache(); gc.collect()

    if not predictions:
        print("No models found. Exiting.")
        return

    # -----------------------------------------------------------------------
    # FIGURE 1: Extrapolation Error vs Time (body = Earth, index 3)
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, pred in predictions.items():
        err = np.linalg.norm(pred[:, BODY_IDX, :3] - true_st[:, BODY_IDX, :3], axis=-1)
        ls = "--" if name == "HPINN" else "-"
        lw = 2.0 if name == "PINN" else 1.5
        ax.plot(t_years, err, color=MODEL_COLORS[name], label=name, alpha=0.85, linewidth=lw, linestyle=ls)

    ax.axvline(t_years[split_idx], color="black", linestyle="--", linewidth=1.2,
               label="Training boundary")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1e7],
                     t_years[split_idx], t_years[-1],
                     alpha=0.06, color="black")
    ax.set_yscale("log")
    ax.set_xlabel("Time (years from epoch J2000)")
    ax.set_ylabel("Position Error (km)  [log scale]")
    ax.set_title(f"Long-Term Extrapolation Error — {BODY_NAME.capitalize()} (body #{BODY_IDX})")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "fig1_extrapolation_error.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")

    # -----------------------------------------------------------------------
    # FIGURE 2: Orbital Energy Drift — Δ E / E₀  (body = Earth)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, pred in predictions.items():
        E0   = true_E[:, BODY_IDX]
        E_p  = specific_energy(pred)[:, BODY_IDX]
        dE   = (E_p - E0) / (np.abs(E0) + 1e-12)
        ls = "--" if name == "HPINN" else "-"
        lw = 2.0 if name == "PINN" else 1.5
        ax.plot(t_years, dE, color=MODEL_COLORS[name], label=name, alpha=0.85, linewidth=lw, linestyle=ls)

    ax.axvline(t_years[split_idx], color="black", linestyle="--", linewidth=1.2,
               label="Training boundary")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (years from epoch J2000)")
    ax.set_ylabel(r"Relative Energy Drift  $\Delta E / E_0$")
    ax.set_title(f"Dynamical Consistency: Specific Orbital Energy — {BODY_NAME.capitalize()}")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "fig2_energy_drift.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")

    # -----------------------------------------------------------------------
    # FIGURE 3: Angular Momentum Conservation — |L| / |L₀|  (body = Earth)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    L0 = true_L[:, BODY_IDX]

    for name, pred in predictions.items():
        L_p  = angular_momentum_magnitude(pred)[:, BODY_IDX]
        dL   = (L_p - L0) / (L0 + 1e-12)
        ls = "--" if name == "HPINN" else "-"
        lw = 2.0 if name == "PINN" else 1.5
        ax.plot(t_years, dL, color=MODEL_COLORS[name], label=name, alpha=0.85, linewidth=lw, linestyle=ls)

    ax.axvline(t_years[split_idx], color="black", linestyle="--", linewidth=1.2,
               label="Training boundary")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (years from epoch J2000)")
    ax.set_ylabel(r"Relative Angular Momentum Drift  $\Delta |L| / |L_0|$")
    ax.set_title(f"Dynamical Consistency: Angular Momentum — {BODY_NAME.capitalize()}")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "fig3_angular_momentum.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")

    # -----------------------------------------------------------------------
    # FIGURE 4: Radial Distance Error — |r_pred| - |r_true| (body = Earth)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, pred in predictions.items():
        r_true = np.linalg.norm(true_st[:, BODY_IDX, :3], axis=-1)
        r_pred = np.linalg.norm(pred[:, BODY_IDX, :3], axis=-1)
        err_radial = r_pred - r_true
        ls = "--" if name == "HPINN" else "-"
        lw = 2.0 if name == "PINN" else 1.5
        ax.plot(t_years, err_radial, color=MODEL_COLORS[name], label=name, alpha=0.85, linewidth=lw, linestyle=ls)

    ax.axvline(t_years[split_idx], color="black", linestyle="--", linewidth=1.2,
               label="Training boundary")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (years from epoch J2000)")
    ax.set_ylabel("Radial Error $\Delta |r|$ (km)")
    ax.set_title(f"Orbital Shape Stability: Radial Distance Error — {BODY_NAME.capitalize()}")
    
    # Non usiamo il log qui perché l'errore può essere negativo (orbita più stretta)
    # Ma mettiamo un limite simmetrico per vedere bene la HPINN
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "fig4_radial_error.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")

    print("\n✅  Paper 1 — all 4 figures saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()
