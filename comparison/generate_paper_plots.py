#!/usr/bin/env python3
import json
import sys
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent
COMP_DIR = WORKSPACE / "comparison"
PLOTS_DIR = COMP_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Add modules to path for imports
sys.path.append(str(WORKSPACE / "HPINN" / "src"))

from solsys_emulator.de440_dataset import load_dataset
from solsys_emulator.model import EmulatorModel, ModelConfig
from solsys_emulator.viz_3d import plot_scene

DATASET_PATH = WORKSPACE / "data" / "dataset_de440.npz"

MLP_SUMMARY = WORKSPACE / "MLP" / "artifacts" / "mlp_train_summary.json"
MLP_CKPT = WORKSPACE / "MLP" / "artifacts" / "emulator_stage1.pt"

PINN_SUMMARY = WORKSPACE / "PINN" / "training_summary.json"
PINN_CKPT = WORKSPACE / "PINN" / "artifacts" / "final_checkpoint.pt"

HPINN_SUMMARY = WORKSPACE / "HPINN" / "training_summary.json"
HPINN_CKPT = WORKSPACE / "HPINN" / "artifacts" / "final_checkpoint.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
    })

def plot_paper2_benchmark_barchart():
    models = ["MLP (Data-Driven)", "PINN (Physics-Informed)", "HPINN (Hybrid Physics)"]
    rmses = []
    for p in [MLP_SUMMARY, PINN_SUMMARY, HPINN_SUMMARY]:
        if p.exists():
            data = json.loads(p.read_text())
            rmses.append(data.get("best_val_position_rmse_km", 0))
        else:
            rmses.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    bars = ax.bar(models, rmses, color=colors, width=0.6)
    
    ax.set_yscale("log")
    ax.set_ylabel("Validation Position RMSE (km) [Log Scale]")
    ax.set_title("Solar System Ephemeris Emulation Benchmark (DE440)", pad=20)
    
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.annotate(f"{int(rmse):,} km",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    out_path = PLOTS_DIR / "paper2_benchmark_rmse.pdf"
    plt.savefig(out_path)
    plt.close()

def plot_paper3_staged_optimization_history():
    hpinn_dir = WORKSPACE / "HPINN" / "artifacts"
    stages = [
        ("Coarse", hpinn_dir / "history_coarse.json"),
        ("Refine", hpinn_dir / "history_refine.json"),
        ("Physics", hpinn_dir / "history_physics.json")
    ]
    val_rmse = []
    epochs_so_far = 0
    boundaries = []
    
    for name, p in stages:
        if p.exists():
            hist = json.loads(p.read_text())
            rmse = hist.get("val_pos_rmse_km", [])
            val_rmse.extend(rmse)
            epochs_so_far += len(rmse)
            boundaries.append((epochs_so_far, name))
            
    if not val_rmse:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(val_rmse, color="#1f77b4", label="Position RMSE (km)", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Validation RMSE (km)", color="#1f77b4")
    
    prev_b = 0
    for b, name in boundaries:
        ax1.axvline(x=b, color='k', linestyle='--', alpha=0.5)
        ax1.text(prev_b + (b - prev_b)/2, max(val_rmse)*0.8, name, 
                 ha='center', va='top', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        prev_b = b

    plt.title("HPINN: Multi-Stage Training Convergence", pad=15)
    plt.tight_layout()
    out_path = PLOTS_DIR / "paper3_staged_optimization.pdf"
    plt.savefig(out_path)
    plt.close()

def calculate_energy(states_km_kms):
    """Calcola l'energia specifica approssimata per le derive"""
    pos = states_km_kms[..., :3]
    vel = states_km_kms[..., 3:]
    v_sq = np.sum(vel**2, axis=-1)
    mu_sun = 132712440041.9394
    r = np.linalg.norm(pos, axis=-1)
    energy = 0.5 * v_sq - (mu_sun / r)
    return energy

def extended_plots():
    if not DATASET_PATH.exists():
        print("Dataset non trovato.")
        return
        
    dataset = load_dataset(DATASET_PATH)
    times_sec = dataset["times_seconds"]
    true_states = dataset["states"]
    bodies = dataset["bodies"]
    
    # Calcolo time_norm basato sul dataset (come faceva train.py)
    t_min, t_max = np.min(times_sec), np.max(times_sec)
    time_mean = (t_max + t_min) / 2.0
    time_std = (t_max - t_min) / 2.0
    t_norm = (times_sec - time_mean) / time_std
    t_norm_t = torch.from_numpy(t_norm).float()
    
    models_to_test = [
        ("MLP", MLP_CKPT),
        ("PINN", PINN_CKPT),
        ("HPINN", HPINN_CKPT)
    ]
    
    results = {}
    delta_rs = {}
    
    print("Avvio inferenza sui modelli...")
    for name, path in models_to_test:
        if not path.exists():
            continue
            
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg_dict = ckpt.get("model_kwargs", ckpt.get("model_config", {}))
        
        # Gestione retro-compatibilità delle kwargs nei checkpoint
        if name == "PINN":
            cfg_dict["state_mode"] = "position_only"
        elif name == "HPINN":
            cfg_dict["state_mode"] = "position_only"
            cfg_dict["hybrid_correction"] = True
        
        # MLP era hardcoded con residual blocks nel suo sorgente vecchio
        if name == "MLP":
            cfg_dict["backbone_type"] = "residual"
            
        cfg = ModelConfig(**cfg_dict)
        model = EmulatorModel(**cfg.to_kwargs()).to(DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", {}))
        
        # Mappatura chiavi legacy (MLP usava linear1/linear2, HPINN usa fc1/fc2 nel ResidualBlock)
        if name == "MLP":
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace(".linear1.", ".fc1.").replace(".linear2.", ".fc2.")
                
                # Fix offset: MLP ha i blocchi da 2 a 9, HPINN li ha da 1 a 8
                if "backbone." in new_k:
                    parts = new_k.split(".")
                    try:
                        idx = int(parts[1])
                        if idx >= 2:
                            parts[1] = str(idx - 1)
                            new_k = ".".join(parts)
                    except ValueError:
                        pass
                        
                new_state_dict[new_k] = v
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict)
        model.eval()
        
        if "scaler_mean" in ckpt:
            scaler_mean = torch.from_numpy(ckpt["scaler_mean"]).float().to(DEVICE)
            scaler_std = torch.from_numpy(ckpt["scaler_std"]).float().to(DEVICE)
        else:
            scaler_mean = torch.tensor(ckpt["scaler"]["mean"]).float().to(DEVICE)
            scaler_std = torch.tensor(ckpt["scaler"]["std"]).float().to(DEVICE)
        
        if cfg.state_mode == "position_only":
            from torch import vmap
            from torch.func import jacfwd
            def _single_time_pos_norm(t_s_1d: torch.Tensor) -> torch.Tensor:
                return model(t_s_1d.unsqueeze(0)).squeeze(0)
            model._vmap_fns = {"vel": vmap(jacfwd(_single_time_pos_norm))}
            
        with torch.no_grad():
            chunk_size = 500
            n_chunks = int(np.ceil(len(t_norm_t) / chunk_size))
            all_pos = []
            all_vel = []
            all_delta = []
            
            for i in range(n_chunks):
                t_chunk = t_norm_t[i*chunk_size : (i+1)*chunk_size].to(DEVICE)
                
                if cfg.state_mode == "full":
                    states = model(t_chunk)
                    # Un-normalize 6D state
                    states_unnorm = states * scaler_std + scaler_mean
                    all_pos.append(states_unnorm[..., :3].cpu())
                    all_vel.append(states_unnorm[..., 3:].cpu())
                else:
                    if cfg.hybrid_correction:
                        pos_norm, base_out, delta_out = model.forward_components(t_chunk)
                        all_delta.append((delta_out * scaler_std[..., :3]).cpu())
                    else:
                        pos_norm = model(t_chunk)
                        
                    vel_norm = model._vmap_fns["vel"](t_chunk)
                    
                    # Un-normalize (vel_norm deve essere de-scalata col tempo)
                    pos = pos_norm * scaler_std[..., :3] + scaler_mean[..., :3]
                    vel = vel_norm * (scaler_std[..., :3] / time_std)
                    all_pos.append(pos.cpu())
                    all_vel.append(vel.cpu())
            
            pred_states = torch.cat([torch.cat(all_pos, dim=0), torch.cat(all_vel, dim=0)], dim=-1).numpy()
            results[name] = pred_states
            if all_delta:
                delta_rs[name] = torch.cat(all_delta, dim=0).numpy()
                
        del model
        torch.cuda.empty_cache()
        gc.collect()

    t_years = times_sec / (365.25 * 86400)
    split_idx = int(len(times_sec) * 0.9)
    
    # PAPER 1: ERRORE vs TEMPO
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in results:
        err = np.linalg.norm(results[name][:, 0, :3] - true_states[:, 0, :3], axis=-1)
        ax.plot(t_years, err, label=name, alpha=0.8)
    ax.axvline(t_years[split_idx], color='k', linestyle='--', label='End of Training')
    ax.set_yscale('log')
    ax.set_ylabel("Position Error (km)")
    ax.set_xlabel("Time (Years from Epoch)")
    ax.set_title("Long-Term Extrapolation Error (Body 0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "paper1_extrapolation_error.pdf")
    plt.close()
    
    # PAPER 1: ENERGY DRIFT
    fig, ax = plt.subplots(figsize=(10, 5))
    true_energy = calculate_energy(true_states)
    for name in results:
        en = calculate_energy(results[name])
        delta_en = (en[:, 0] - true_energy[:, 0]) / np.abs(true_energy[:, 0] + 1e-9)
        ax.plot(t_years, delta_en, label=name, alpha=0.8)
    ax.set_ylabel("Relative Energy Drift $\Delta E / E_0$")
    ax.set_xlabel("Time (Years from Epoch)")
    ax.set_title("Dynamical Consistency: Energy Drift")
    ax.set_ylim(-1, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "paper1_energy_drift.pdf")
    plt.close()
    
    # PAPER 4: HYBRID RESIDUAL
    if "HPINN" in delta_rs:
        dr = delta_rs["HPINN"]
        dr_mag = np.linalg.norm(dr, axis=-1)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in [0, 4, 8]: 
            ax.plot(t_years, dr_mag[:, i], label=bodies[i], alpha=0.8)
        ax.set_ylabel("Residual Magnitude $|\Delta r|$ (km)")
        ax.set_xlabel("Time (Years from Epoch)")
        ax.set_title("HPINN Learned Residual over Time")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "paper4_residual_magnitude.pdf")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in [0, 4, 8]: 
            signal = dr_mag[:, i] - np.mean(dr_mag[:, i])
            fft_vals = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(len(signal), d=1.0)
            ax.plot(freqs[1:], fft_vals[1:], label=bodies[i], alpha=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel("FFT Amplitude")
        ax.set_xlabel("Frequency (1/Days)")
        ax.set_title("Spectral Analysis of Learned Residual")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "paper4_residual_fft.pdf")
        plt.close()
        
    # PAPER 2: 3D ORBIT SCENE (HTML)
    if "HPINN" in results:
        # Pochi punti per l'html (ultimi anni)
        sub_states = results["HPINN"][-1000:]
        sub_true = true_states[-1000:]
        traj_hpinn = {bodies[i]: sub_states[:, i, :3] for i in range(len(bodies))}
        traj_true = {bodies[i]: sub_true[:, i, :3] for i in range(len(bodies))}
        states_0 = {bodies[i]: sub_states[0, i, :3] for i in range(len(bodies))}
        
        fig = plot_scene(states_at_t=states_0, trajectories=traj_hpinn, reference_trajectories=traj_true, reference_label="DE440")
        fig.write_html(str(PLOTS_DIR / "paper2_orbits_3d.html"))

if __name__ == "__main__":
    setup_plot_style()
    print("Generazione Benchmark (Paper 2 e 3)...")
    plot_paper2_benchmark_barchart()
    plot_paper3_staged_optimization_history()
    print("Esecuzione Inferenza e Plot 3D/Energetici (Paper 1, 2, 4)...")
    extended_plots()
    print("✅ Tutte le analisi e i plot sono stati generati.")
