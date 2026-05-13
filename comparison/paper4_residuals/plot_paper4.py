#!/usr/bin/env python3
"""
Paper 4: Physics-Residual Hybrid Modeling
Focuses on the Delta-r correction term learned by the HPINN.
"""
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths & Setup
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
HPINN_PATH   = WORKSPACE / "HPINN" / "artifacts" / "final_checkpoint.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.titlesize":   13,
        "figure.dpi":       200,
    })

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_style()
    
    print("Loading dataset and HPINN model…")
    dataset = load_dataset(DATASET_PATH)
    bodies  = list(dataset["bodies"])
    times_sec = dataset["times_seconds"]
    
    t_mean = float(np.mean(times_sec))
    t_std  = float(np.std(times_sec))
    t_norm = (times_sec - t_mean) / t_std
    t_norm_t = torch.from_numpy(t_norm).float().to(DEVICE)

    # Load HPINN
    ckpt = torch.load(HPINN_PATH, map_location=DEVICE, weights_only=False)
    cfg_dict = dict(ckpt.get("model_kwargs", ckpt.get("model_config", {})))
    cfg_dict["state_mode"] = "position_only"
    cfg_dict["hybrid_correction"] = True
    
    model = EmulatorModel(**ModelConfig(**cfg_dict).to_kwargs()).to(DEVICE)
    model.load_state_dict(dict(ckpt.get("model_state_dict", ckpt.get("model_state", {}))))
    model.eval()

    if "scaler_mean" in ckpt:
        s_mean = torch.from_numpy(ckpt["scaler_mean"]).float().to(DEVICE)
        s_std  = torch.from_numpy(ckpt["scaler_std"]).float().to(DEVICE)
    else:
        s_mean = torch.tensor(ckpt["scaler"]["mean"]).float().to(DEVICE)
        s_std  = torch.tensor(ckpt["scaler"]["std"]).float().to(DEVICE)

    print("Extracting physical and residual components…")
    all_phys, all_delta = [], []
    chunk = 1000
    
    with torch.no_grad():
        for i in range(int(np.ceil(len(t_norm_t) / chunk))):
            t_ch = t_norm_t[i*chunk:(i+1)*chunk]
            r_phys_norm, delta_r_norm, _ = model.forward_components(t_ch)
            r_phys = r_phys_norm * s_std[..., :3] + s_mean[..., :3]
            delta_r = delta_r_norm * s_std[..., :3]
            all_phys.append(r_phys.cpu())
            all_delta.append(delta_r.cpu())

    phys_pos = torch.cat(all_phys, 0).numpy()
    delta_pos = torch.cat(all_delta, 0).numpy()
    t_years = times_sec / (365.25 * 86400)

    # FIGURE 1: Residual Magnitude
    fig, ax = plt.subplots(figsize=(10, 6))
    TARGET_BODIES = [1, 3, 5, 6]
    for b_idx in TARGET_BODIES:
        mag = np.linalg.norm(delta_pos[:, b_idx], axis=-1)
        ax.plot(t_years, mag, label=f"{bodies[b_idx].capitalize()}", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Residual Magnitude $|\Delta r|$ (km)")
    ax.set_title("Learned Corrections to Newtonian Physics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_residual_magnitude.pdf")
    plt.close()

    # FIGURE 2: Residual Ratio
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_phys = np.mean(np.linalg.norm(phys_pos, axis=-1), axis=0)
    mean_delta = np.mean(np.linalg.norm(delta_pos, axis=-1), axis=0)
    x = np.arange(len(bodies))
    ax.bar(x, mean_delta, color="#bcbd22", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bodies], rotation=45)
    ax.set_yscale("log")
    ax.set_ylabel("Mean Delta (km)")
    ax.set_title("Average Magnitude of Learned Residuals")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_residual_ratio.pdf")
    plt.close()

    print(f"\n✅ Paper 4 figures saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
