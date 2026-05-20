#!/usr/bin/env python3
"""
Paper 2: Comparing Data-Driven and Physics-Informed Neural Models for Solar-System Orbit Emulation
Benchmark figures:
  1. Global RMSE Comparison (Bar Chart)
  2. Per-Body RMSE Heatmap
  3. Interactive 3D Orbit Scene (Plotly)
"""
import gc
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go

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
from solsys_emulator.viz_3d import plot_scene  # Reuse the 3D plotting logic

DATASET_PATH = WORKSPACE / "data" / "dataset_de440.npz"

CHECKPOINTS = {
    "MLP":   WORKSPACE / "MLP"  / "artifacts" / "emulator_stage1.pt",
    "PINN":  WORKSPACE / "PINN" / "artifacts_768x8" / "final_checkpoint.pt" / "checkpoint_physics.pt",
    "HPINN": WORKSPACE / "HPINN"/ "artifacts" / "final_checkpoint.pt",
}

MODEL_COLORS = {
    "MLP":   "#d62728",
    "PINN":  "#1f77b4",
    "HPINN": "#2ca02c",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.titlesize":   13,
        "figure.dpi":       200,
    })

def load_model_helper(name: str, path: Path):
    if not path.exists(): return None, None, None, None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg_dict = dict(ckpt.get("model_kwargs", ckpt.get("model_config", {})))
    if name in ("MLP", "PINN", "HPINN"): cfg_dict["state_mode"] = "position_only"
    if name == "HPINN": cfg_dict["hybrid_correction"] = True
    
    cfg = ModelConfig(**cfg_dict)
    model = EmulatorModel(**cfg.to_kwargs()).to(DEVICE)
    
    state_dict = dict(ckpt.get("model_state_dict", ckpt.get("model_state", {})))
    model.load_state_dict(state_dict)
    model.eval()

    if "scaler_mean" in ckpt:
        s_mean = torch.from_numpy(ckpt["scaler_mean"]).float().to(DEVICE)
        s_std  = torch.from_numpy(ckpt["scaler_std"]).float().to(DEVICE)
    else:
        s_mean = torch.tensor(ckpt["scaler"]["mean"]).float().to(DEVICE)
        s_std  = torch.tensor(ckpt["scaler"]["std"]).float().to(DEVICE)
    return model, cfg, s_mean, s_std

@torch.no_grad()
def predict_states(name, model, cfg, s_mean, s_std, t_norm_t, time_std):
    chunk = 1000
    all_pos = []
    
    if cfg.state_mode == "position_only":
        from torch import vmap
        from torch.func import jacfwd
        _vmap_pos = vmap(lambda t: model(t.unsqueeze(0)).squeeze(0))

    for i in range(int(np.ceil(len(t_norm_t) / chunk))):
        t_ch = t_norm_t[i*chunk:(i+1)*chunk].to(DEVICE)
        if cfg.state_mode == "full":
            out = model(t_ch) * s_std + s_mean
            all_pos.append(out[..., :3].cpu())
        else:
            if name == "HPINN":
                pos_norm, _, _ = model.forward_components(t_ch)
            else:
                pos_norm = model(t_ch)
            pos = pos_norm * s_std[..., :3] + s_mean[..., :3]
            all_pos.append(pos.cpu())
    return torch.cat(all_pos, 0).numpy()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_style()
    print("Loading dataset…")
    dataset = load_dataset(DATASET_PATH)
    true_st = dataset["states"][..., :3] # Only pos
    bodies  = list(dataset["bodies"])
    times_sec = dataset["times_seconds"]
    
    t_mean = float(np.mean(times_sec))
    t_std  = float(np.std(times_sec))
    t_norm = (times_sec - t_mean) / t_std
    t_norm_t = torch.from_numpy(t_norm).float()

    results = {}
    per_body_rmse = {} # model -> array of RMSE per body

    for name, path in CHECKPOINTS.items():
        print(f"Running Benchmark: {name}…")
        model, cfg, s_mean, s_std = load_model_helper(name, path)
        if model is None: continue
        
        pred = predict_states(name, model, cfg, s_mean, s_std, t_norm_t, t_std)
        results[name] = pred
        
        # Calculate RMSE per body
        diff = pred - true_st
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=-1), axis=0))
        per_body_rmse[name] = rmse
        
        del model; torch.cuda.empty_cache(); gc.collect()

    # -----------------------------------------------------------------------
    # FIGURE 1: Global Mean RMSE (Bar Chart)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(per_body_rmse.keys())
    means = [np.mean(per_body_rmse[n]) for n in names]
    
    bars = ax.bar(names, means, color=[MODEL_COLORS[n] for n in names], alpha=0.8, edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Global Mean Position RMSE (km) [log]")
    ax.set_title("Global Benchmark: MLP vs PINN vs HPINN")
    
    # Add values on top
    max_height = max(means)
    ax.set_ylim(min(means)*0.5, max_height * 4) # Spazio extra in cima ridotto
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.25,
                f'{height:,.0f} km', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_global_rmse.pdf")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 2: Per-Body Heatmap
    # -----------------------------------------------------------------------
    data_matrix = np.array([per_body_rmse[n] for n in names]) # [Models, Bodies]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.log10(data_matrix), cmap="YlOrRd", aspect="auto")
    
    ax.set_xticks(np.arange(len(bodies)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels([b.capitalize() for b in bodies])
    ax.set_yticklabels(names)
    
    # Rotate body labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    log_data = np.log10(data_matrix)
    vmin, vmax = log_data.min(), log_data.max()
    
    for i in range(len(names)):
        for j in range(len(bodies)):
            val = data_matrix[i, j]
            
            # Format text dynamically
            if val >= 1e9:
                text_val = f"{val/1e9:.1f}B"
            elif val >= 1e6:
                text_val = f"{val/1e6:.1f}M"
            else:
                text_val = f"{val/1000:.0f}k"
                
            # Better contrast based on normalized log value
            norm_val = (log_data[i, j] - vmin) / (vmax - vmin + 1e-9)
            color = "white" if norm_val > 0.6 else "black"
            
            ax.text(j, i, text_val, ha="center", va="center", 
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title("Per-Body Position RMSE (km)")
    fig.colorbar(im, label="log10 RMSE")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_per_body_heatmap.pdf")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 3: Interactive 3D Scenes (HTML)
    # -----------------------------------------------------------------------
    print("Generating 3D Scenes…")
    
    def create_3d_plot(start_idx, end_idx, title, filename):
        fig_3d = go.Figure()
        BODIES_TO_PLOT = [3] # Solo la Terra
        
        for b_idx in BODIES_TO_PLOT:
            b_name = bodies[b_idx].capitalize()
            
            # Use slice if end_idx is None
            if end_idx is None:
                gt = true_st[start_idx:, b_idx]
            else:
                gt = true_st[start_idx:end_idx, b_idx]
                
            fig_3d.add_trace(go.Scatter3d(
                x=gt[:, 0], y=gt[:, 1], z=gt[:, 2],
                mode='lines', line=dict(color='black', width=2, dash='dash'),
                name=f"{b_name} (Ground Truth)"
            ))
            
            for n in names:
                if end_idx is None:
                    p = results[n][start_idx:, b_idx]
                else:
                    p = results[n][start_idx:end_idx, b_idx]
                    
                fig_3d.add_trace(go.Scatter3d(
                    x=p[:, 0], y=p[:, 1], z=p[:, 2],
                    mode='lines', line=dict(color=MODEL_COLORS[n], width=4),
                    name=f"{b_name} ({n})"
                ))

        fig_3d.update_layout(
            title=dict(text=title, font=dict(size=32)),
            scene=dict(
                xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)',
                xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
                yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
                zaxis=dict(title_font=dict(size=24), tickfont=dict(size=20))
            ),
            legend=dict(font=dict(size=26), itemsizing='constant'),
            margin=dict(l=0, r=0, b=0, t=80)
        )
        fig_3d.write_html(PLOTS_DIR / filename)

    def create_error_3d_plot(start_idx, end_idx, title, filename):
        fig_3d = go.Figure()
        BODIES_TO_PLOT = [3] # Solo Terra
        
        for b_idx in BODIES_TO_PLOT:
            b_name = bodies[b_idx].capitalize()
            
            # Origin is the Ground Truth in error space
            fig_3d.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers', marker=dict(color='black', size=8, symbol='cross'),
                name=f"{b_name} (Ground Truth Origin)"
            ))
            
            for n in names:
                if end_idx is None:
                    p = results[n][start_idx:, b_idx]
                    gt = true_st[start_idx:, b_idx]
                else:
                    p = results[n][start_idx:end_idx, b_idx]
                    gt = true_st[start_idx:end_idx, b_idx]
                    
                err = p - gt
                fig_3d.add_trace(go.Scatter3d(
                    x=err[:, 0], y=err[:, 1], z=err[:, 2],
                    mode='lines', line=dict(color=MODEL_COLORS[n], width=4),
                    name=f"{b_name} Error ({n})"
                ))

        fig_3d.update_layout(
            title=dict(text=title, font=dict(size=32)),
            scene=dict(
                xaxis_title='Delta X (km)', yaxis_title='Delta Y (km)', zaxis_title='Delta Z (km)',
                xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
                yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
                zaxis=dict(title_font=dict(size=24), tickfont=dict(size=20))
            ),
            legend=dict(font=dict(size=26), itemsizing='constant'),
            margin=dict(l=0, r=0, b=0, t=80)
        )
        fig_3d.write_html(PLOTS_DIR / filename)

    # 3A: Interpolation (2020)
    # 2010 to 2020 is 10 years. 10 * 365.25 * 8 = ~29220 steps
    start_2020 = int(10 * 365.25 * 8)
    end_2020 = int(11 * 365.25 * 8)
    create_3d_plot(start_2020, end_2020, 
                   "Orbital Trajectory Benchmark: Interpolation Phase (Year 2020, inside training set)", 
                   "fig3a_interpolation_orbits.html")

    # 3B: Extrapolation (2029)
    # Last 365 days of dataset
    start_2029 = -int(365 * 8)
    end_2029 = None
    create_3d_plot(start_2029, end_2029, 
                   "Orbital Trajectory Benchmark: Extrapolation Phase (Year 2029, 1 year into unknown future)", 
                   "fig3b_extrapolation_orbits.html")

    # 4: Close-up Detail (First 180 days of 2029)
    start_detail = -int(365 * 8)
    end_detail = -int((365 - 180) * 8)
    create_3d_plot(start_detail, end_detail, 
                   "Close-up Detail: Orbit Divergence in Extrapolation (180-day arc)", 
                   "fig4_orbit_detail.html")
                   
    # 5: Error Space 3D (Extrapolation phase)
    create_error_3d_plot(start_2029, end_2029,
                         "3D Error Space (Delta XYZ): Divergence from Ground Truth (Extrapolation)",
                         "fig5_error_space_3d.html")

    # -----------------------------------------------------------------------
    # FIGURE 6: Static 2D Close-up Zoom (For LaTeX PDF)
    # -----------------------------------------------------------------------
    print("Generating Static 2D Close-up (PDF)…")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 120-day arc
    start_2d = -int(365 * 8)
    end_2d = -int((365 - 120) * 8)
    b_idx = 3 # Earth
    
    gt_2d = true_st[start_2d:end_2d, b_idx]
    ax.plot(gt_2d[:, 0], gt_2d[:, 1], 'k--', linewidth=2, label="Ground Truth")
    
    for n in names:
        p_2d = results[n][start_2d:end_2d, b_idx]
        # Linee più sottili per far vedere meglio lo scollamento
        ax.plot(p_2d[:, 0], p_2d[:, 1], color=MODEL_COLORS[n], linewidth=1.5, alpha=0.9, label=n)
        
    # Zoom molto stretto (finestra di +/- 250.000 km) sull'ultimo punto dell'arco
    x_center = gt_2d[-1, 0]
    y_center = gt_2d[-1, 1]
    window = 250_000
    
    ax.set_xlim(x_center - window, x_center + window)
    ax.set_ylim(y_center - window, y_center + window)
    
    ax.set_aspect('equal')
    ax.set_title("Close-up Detail: Orbit Divergence (Earth 2D Projection)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig6_orbit_detail_2d.pdf")
    plt.close()
    
    print(f"\n✅ Paper 2 figures saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
