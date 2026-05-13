#!/usr/bin/env python3
"""
Paper 3: Multi-Stage Curriculum Learning for Physics-Informed Orbital Emulators
Plots the training history across Coarse, Refine, and Physics stages.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
COMP_DIR    = SCRIPT_DIR.parent
WORKSPACE   = COMP_DIR.parent
PLOTS_DIR   = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# We focus on the HPINN and PINN 768x8 runs
HPINN_HIST_DIR = WORKSPACE / "HPINN" / "artifacts"
PINN_HIST_DIR  = WORKSPACE / "PINN" / "artifacts_768x8" / "final_checkpoint.pt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_history(base_path, stages=["coarse", "refine", "physics"]):
    full_hist = {"train_loss": [], "val_pos_rmse_km": [], "stage_boundaries": []}
    current_offset = 0
    
    for stage in stages:
        h_path = base_path / f"history_{stage}.json"
        if not h_path.exists():
            print(f"  [SKIP] Stage {stage} not found at {h_path}")
            continue
            
        with open(h_path, "r") as f:
            h = json.load(f)
            
        full_hist["train_loss"].extend(h.get("train_loss", []))
        full_hist["val_pos_rmse_km"].extend(h.get("val_pos_rmse_km", []))
        
        current_offset += len(h.get("train_loss", []))
        full_hist["stage_boundaries"].append(current_offset)
        
    return full_hist

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
    
    print("Loading histories…")
    hpinn_hist = load_history(HPINN_HIST_DIR)
    pinn_hist  = load_history(PINN_HIST_DIR)

    # -----------------------------------------------------------------------
    # FIGURE 1: Multi-Stage Training Progress (RMSE)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot PINN
    if pinn_hist["val_pos_rmse_km"]:
        ax.plot(pinn_hist["val_pos_rmse_km"], label="PINN (768x8)", color="#1f77b4", alpha=0.8, linewidth=1.5)
        # Vertical lines for PINN stages
        for b in pinn_hist["stage_boundaries"][:-1]:
            ax.axvline(b, color="#1f77b4", linestyle=":", alpha=0.5)

    # Plot HPINN
    if hpinn_hist["val_pos_rmse_km"]:
        ax.plot(hpinn_hist["val_pos_rmse_km"], label="HPINN (768x8)", color="#2ca02c", alpha=0.8, linewidth=2)
        # Vertical lines for HPINN stages
        for b in hpinn_hist["stage_boundaries"][:-1]:
            ax.axvline(b, color="#2ca02c", linestyle="--", alpha=0.5)

    # Stage Labels
    stages = ["COARSE", "REFINE", "PHYSICS"]
    # Assuming both have similar stage lengths for label placement
    boundaries = hpinn_hist["stage_boundaries"] if hpinn_hist["stage_boundaries"] else pinn_hist["stage_boundaries"]
    
    start = 0
    for i, end in enumerate(boundaries):
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1]*0.5, stages[i], 
                horizontalalignment='center', fontweight='bold', fontsize=12, alpha=0.3)
        start = end

    ax.set_yscale("log")
    ax.set_xlabel("Total Epochs (Cumulative)")
    ax.set_ylabel("Validation Position RMSE (km)")
    ax.set_title("Curriculum Learning Progress: MLP vs PINN vs HPINN")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_training_curriculum.pdf")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 2: Stage-wise Best RMSE Comparison
    # -----------------------------------------------------------------------
    # Extract best RMSE per stage
    def get_stage_bests(base_path):
        bests = []
        for s in ["coarse", "refine", "physics"]:
            p = base_path / f"history_{s}.json"
            if p.exists():
                with open(p, "r") as f:
                    h = json.load(f)
                    bests.append(np.min(h["val_pos_rmse_km"]))
            else:
                bests.append(None)
        return bests

    hpinn_bests = get_stage_bests(HPINN_HIST_DIR)
    pinn_bests  = get_stage_bests(PINN_HIST_DIR)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, [b for b in pinn_bests if b is not None], width, label='PINN', color="#1f77b4", alpha=0.7)
    ax.bar(x + width/2, [b for b in hpinn_bests if b is not None], width, label='HPINN', color="#2ca02c", alpha=0.7)
    
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Stage 1: Coarse", "Stage 2: Refine", "Stage 3: Physics"])
    ax.set_ylabel("Best Position RMSE (km)")
    ax.set_title("Model Performance Evolution per Stage")
    ax.legend()

    # Add text values
    for i, val in enumerate(pinn_bests):
        if val: ax.text(i - width/2, val, f"{val:,.0f}", ha='center', va='bottom', fontsize=9)
    for i, val in enumerate(hpinn_bests):
        if val: ax.text(i + width/2, val, f"{val:,.0f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_stage_comparison.pdf")
    plt.close()

    print(f"\n✅ Paper 3 figures saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
