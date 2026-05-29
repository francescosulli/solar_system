#!/usr/bin/env python3
import gc
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.time import Time

SCRIPT_DIR  = Path(__file__).resolve().parent
COMP_DIR    = SCRIPT_DIR.parent.parent
WORKSPACE   = COMP_DIR.parent
PLOTS_DIR   = SCRIPT_DIR

sys.path.append(str(WORKSPACE / "HPINN" / "src"))
from solsys_emulator.de440_dataset import load_dataset, sample_states, load_kernel
from solsys_emulator.time_frames import build_time_grid
from solsys_emulator.model import EmulatorModel, ModelConfig

DATASET_PATH = WORKSPACE / "data" / "dataset_de440.npz"
KERNEL_PATH = WORKSPACE / "data" / "de440.bsp"

CHECKPOINTS = {
    "MLP":   WORKSPACE / "MLP"  / "artifacts" / "emulator_stage1.pt",
    "PINN":  WORKSPACE / "PINN" / "artifacts_768x8" / "final_checkpoint.pt" / "checkpoint_physics.pt",
    "HPINN": WORKSPACE / "HPINN"/ "artifacts" / "final_checkpoint.pt",
}

MODEL_COLORS = {"MLP": "#d62728", "PINN": "#1f77b4", "HPINN": "#2ca02c"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INNER_BODIES = [1, 2, 3, 4, 5] # Mercury, Venus, Earth, Moon, Mars
OUTER_BODIES = [6, 7, 8, 9]    # Jupiter, Saturn, Uranus, Neptune

def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 13, "figure.dpi": 200,
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
    return np.concatenate([pos_km, vel_kms], axis=-1)

def main():
    setup_style()
    print("Loading base dataset (2010-2030)...")
    dataset = load_dataset(DATASET_PATH)
    times_sec = dataset["times_seconds"]
    true_st = dataset["states"]
    bodies = list(dataset["bodies"])
    
    t_mean = float(np.mean(times_sec))
    t_std  = float(np.std(times_sec))
    
    t_norm_base = torch.from_numpy((times_sec - t_mean) / t_std).float()

    print("Running base benchmark...")
    predictions = {}
    for name, path in CHECKPOINTS.items():
        print(f"  {name}...")
        model, cfg, s_mean, s_std = load_model_helper(name, path)
        if model:
            predictions[name] = predict_states(name, model, cfg, s_mean, s_std, t_norm_base, t_std)
        del model; torch.cuda.empty_cache(); gc.collect()

    # 1 & 2. Inner vs Outer & Velocity RMSE
    inner_pos_rmse, outer_pos_rmse = {}, {}
    inner_vel_rmse, outer_vel_rmse = {}, {}
    global_pos_rmse, global_vel_rmse = {}, {}

    for name, pred in predictions.items():
        pos_diff = pred[..., :3] - true_st[..., :3]
        vel_diff = pred[..., 3:] - true_st[..., 3:]
        
        rmse_pos_per_body = np.sqrt(np.mean(np.sum(pos_diff**2, axis=-1), axis=0))
        rmse_vel_per_body = np.sqrt(np.mean(np.sum(vel_diff**2, axis=-1), axis=0))
        
        global_pos_rmse[name] = np.mean(rmse_pos_per_body)
        global_vel_rmse[name] = np.mean(rmse_vel_per_body)
        
        inner_pos_rmse[name] = np.mean(rmse_pos_per_body[INNER_BODIES])
        outer_pos_rmse[name] = np.mean(rmse_pos_per_body[OUTER_BODIES])
        inner_vel_rmse[name] = np.mean(rmse_vel_per_body[INNER_BODIES])
        outer_vel_rmse[name] = np.mean(rmse_vel_per_body[OUTER_BODIES])

    # Table 1: Global
    with open(PLOTS_DIR / "tables.md", "w") as f:
        f.write("## Global RMSE (Interpolation Phase)\n\n")
        f.write("| Model | Mean Position RMSE [km] | Mean Velocity RMSE [km/s] |\n")
        f.write("|---|---|---|\n")
        for n in ["MLP", "PINN", "HPINN"]:
            f.write(f"| {n} | {global_pos_rmse[n]:,.2f} | {global_vel_rmse[n]:.5f} |\n")
            
        f.write("\n## Inner vs Outer (Position and Velocity)\n\n")
        f.write("| Model | Inner Pos RMSE [km] | Outer Pos RMSE [km] | Inner Vel RMSE [km/s] | Outer Vel RMSE [km/s] |\n")
        f.write("|---|---|---|---|---|\n")
        for n in ["MLP", "PINN", "HPINN"]:
            f.write(f"| {n} | {inner_pos_rmse[n]:,.2f} | {outer_pos_rmse[n]:,.2f} | {inner_vel_rmse[n]:.5f} | {outer_vel_rmse[n]:.5f} |\n")

    # Plot A: Inner vs Outer Position
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.35
    names = ["MLP", "PINN", "HPINN"]
    in_p = [inner_pos_rmse[n] for n in names]
    out_p = [outer_pos_rmse[n] for n in names]
    
    ax.bar(x - width/2, in_p, width, label='Inner Bodies', color='#ff7f0e', edgecolor="black")
    ax.bar(x + width/2, out_p, width, label='Outer Bodies', color='#9467bd', edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Position RMSE (km) [log]")
    ax.set_title("Inner vs Outer Bodies: Position Error")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figA_inner_outer_pos.pdf")
    plt.close()

    # Plot B: Inner vs Outer Velocity
    fig, ax = plt.subplots(figsize=(8, 5))
    in_v = [inner_vel_rmse[n] for n in names]
    out_v = [outer_vel_rmse[n] for n in names]
    ax.bar(x - width/2, in_v, width, label='Inner Bodies', color='#ff7f0e', edgecolor="black")
    ax.bar(x + width/2, out_v, width, label='Outer Bodies', color='#9467bd', edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Velocity RMSE (km/s) [log]")
    ax.set_title("Inner vs Outer Bodies: Velocity Error")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figB_inner_outer_vel.pdf")
    plt.close()

    # 3. Multi-window chronological testing (2031-2038)
    print("Generating extrapolation Ground Truth data from kernel (2031-2038)...")
    kernel = load_kernel(KERNEL_PATH)
    
    windows = [
        ("2031-2032", "2031-01-01T00:00:00", "2033-01-01T00:00:00"),
        ("2033-2034", "2033-01-01T00:00:00", "2035-01-01T00:00:00"),
        ("2035-2036", "2035-01-01T00:00:00", "2037-01-01T00:00:00"),
        ("2037-2038", "2037-01-01T00:00:00", "2039-01-01T00:00:00"),
    ]
    
    chrono_results = {n: [] for n in names}
    chrono_earth = {n: [] for n in names}
    
    for w_name, start, end in windows:
        print(f"  Evaluating {w_name}...")
        w_t_sec, w_t_arr = build_time_grid(start, end, 3 * 3600) # 3 hours
        w_states, _ = sample_states(w_t_arr, bodies=bodies, kernel=kernel)
        
        w_t_norm = torch.from_numpy((w_t_sec - t_mean) / t_std).float()
        
        for name, path in CHECKPOINTS.items():
            model, cfg, s_mean, s_std = load_model_helper(name, path)
            pred = predict_states(name, model, cfg, s_mean, s_std, w_t_norm, t_std)
            
            pos_diff = pred[..., :3] - w_states[..., :3]
            rmse_pos_per_body = np.sqrt(np.mean(np.sum(pos_diff**2, axis=-1), axis=0))
            
            chrono_results[name].append(np.mean(rmse_pos_per_body))
            chrono_earth[name].append(rmse_pos_per_body[3]) # Earth
            del model; torch.cuda.empty_cache(); gc.collect()
            
    # Table 3: Chronological (Global)
    with open(PLOTS_DIR / "tables.md", "a") as f:
        f.write("\n## Chronological Extrapolation (Global Position RMSE)\n\n")
        f.write("| Model | 2031-2032 | 2033-2034 | 2035-2036 | 2037-2038 |\n")
        f.write("|---|---|---|---|---|\n")
        for n in names:
            vals = " | ".join([f"{v:,.0f}" for v in chrono_results[n]])
            f.write(f"| {n} | {vals} |\n")
            
        f.write("\n## Chronological Extrapolation (Earth Position Error)\n\n")
        f.write("| Model | 2031-2032 | 2033-2034 | 2035-2036 | 2037-2038 |\n")
        f.write("|---|---|---|---|---|\n")
        for n in names:
            vals = " | ".join([f"{v:,.0f}" for v in chrono_earth[n]])
            f.write(f"| {n} | {vals} |\n")

    # Plot C: Error growth over windows (Global)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [w[0] for w in windows]
    x_pos = np.arange(len(x_labels))
    
    for n in names:
        ax.plot(x_pos, chrono_results[n], marker='o', color=MODEL_COLORS[n], label=n, linewidth=2)
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_yscale("log")
    ax.set_ylabel("Global Mean Position RMSE (km) [log]")
    ax.set_title("Chronological Generalization (Unseen Future)")
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figC_chronological_growth.pdf")
    plt.close()
    
    # Plot D: Error growth over windows (Earth)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in names:
        ax.plot(x_pos, chrono_earth[n], marker='o', color=MODEL_COLORS[n], label=n, linewidth=2)
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_yscale("log")
    ax.set_ylabel("Earth Position RMSE (km) [log]")
    ax.set_title("Chronological Generalization: Earth Extrapolation")
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figD_chronological_earth.pdf")
    plt.close()
    
    print("\n✅ Done! Tables saved to tables.md and plots saved to PDF.")

if __name__ == "__main__":
    main()
