# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A Solar System simulator built as a research workspace: three neural emulators of Solar System ephemerides, all learning the same map `t -> state of 10 bodies` fitted against JPL DE440.

- [MLP/](MLP/) — purely data-driven baseline
- [PINN/](PINN/) — physics-informed, Newtonian N-body loss
- [HPINN/](HPINN/) — hybrid: physics-first position model + small learned residual `r = r_base + Δr`
- [comparison/](comparison/) — paper figure generation across the three trained checkpoints

**The deliverable is four separate papers**, one per `comparison/paperN_*` folder. The model directories are the experiment infrastructure that feeds them:

| Paper | Folder | Subject |
| --- | --- | --- |
| 1 | `paper1_stability` | Long-term stability: extrapolation error, energy drift, angular momentum conservation |
| 2 | `paper2_benchmark` | MLP vs PINN vs HPINN accuracy benchmark; `aggiunte/` holds later inner-vs-outer and chronological-extrapolation figures |
| 3 | `paper3_optimization` | Multi-stage curriculum learning (coarse → refine → physics) |
| 4 | `paper4_residuals` | The HPINN `Δr` correction term — script exists, `plots/` still empty |

Treat the figures and [comparison/final_values.md](comparison/final_values.md) as the product: changes that invalidate the committed checkpoints or the hardcoded figure paths have publication cost, not just code cost. All three models are deliberately trained at an identical 768×8 backbone so the comparison stays apples-to-apples.

Conceptual background is in [describe.md](describe.md) (Italian); the VRAM/AutoDiff history is in [memory.md](memory.md). Prose docs are in Italian, code and comments in English.

## Environment

One shared venv at the repo root (`.venv`, Python 3.12), targeting a single-GPU DGX Spark (NVIDIA GB10, CUDA 13.0, aarch64). There is no CI, linter, or formatter configured.

```bash
source .venv/bin/activate
# recreate deps if needed:
pip install -r requirements-dgx.txt \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple
```

`pytest` is **not** currently installed in `.venv` — install it before running tests: `pip install -e ".[dev]"` from any of the three project dirs.

## The package-name collision (read this first)

All three projects ship a package literally named `solsys_emulator` under `<project>/src/`, with the same distribution name `solsys-emulator`. Only one can be pip-installed into the shared venv at a time — right now `.venv` has an editable install pointing at **`HPINN/src`**.

Consequences that matter when editing:

- `run_dgx.sh` guards its install with `python -c "import solsys_emulator"`, which always succeeds, so `pip install -e .` never re-runs. Correct imports come instead from each entrypoint doing `sys.path.insert(0, <src>)`.
- **`MLP/MLP_train.py` intentionally imports `HPINN/src`**, not `MLP/src` ([MLP_train.py:26](MLP/MLP_train.py#L26)). The MLP baseline is the HPINN model class configured with `state_mode="position_only"`, `hybrid_correction=False`, and all physics loss weights zeroed. Editing `MLP/src/` does **not** change MLP training — that tree is an older, reduced variant (its `ModelConfig` has no `state_mode`, `backbone_type`, or `hybrid_correction`) and is only exercised by `MLP/tests/`.
- All `comparison/*/plot_paper*.py` scripts likewise `sys.path.append(WORKSPACE / "HPINN" / "src")` and rebuild all three models from that one class.
- So: `HPINN/src/solsys_emulator/` is the de-facto shared library. `PINN/src/` is a genuine fork (used only by `PINN/train.py` + tests) and differs from HPINN only in `model.py`/`train.py`.

## Common commands

Training (each script creates timestamped `checkpoints/<ts>/`, `plots/<ts>/`, `logs/runs/<ts>/` inside its project dir and tees stdout to `run.log`):

```bash
bash PINN/run_dgx.sh [unified|multi]    # also downloads DE440 + builds the shared dataset
bash MLP/run_dgx.sh [--run-stage2]
bash HPINN/run_dgx.sh [unified|multi]
```

Direct invocation, bypassing the wrapper (run from the project dir):

```bash
cd HPINN && python train.py --training-mode unified --device cuda \
    --epochs 1400 --batch-size 768 --collocation-points 192 \
    --dataset-path ../data/dataset_de440.npz --checkpoint-path checkpoints/dev --plots-dir plots/dev
cd PINN && python train.py --training-mode multi --skip-physics-stage   # resumes from stage checkpoints if present
cd MLP  && python MLP_train.py --device cpu --dataset-path ../data/dataset_de440.npz
```

Dataset (one-time; `data/de440.bsp` ~120 MB and `data/dataset_de440.npz` are gitignored):

```bash
cd PINN && python do_setup.py --dataset-path ../data/dataset_de440.npz --dataset-type jpl \
    --start 2010-01-01T00:00:00 --end 2030-01-01T00:00:00 --step-hours 3
cd PINN && python verify_setup.py       # torch/CUDA/driver + module import sanity check
```

Tests (`tests/conftest.py` prepends the project's own `src/`, so the shared venv install is shadowed; `pyproject.toml` also sets `pythonpath = ["src"]`):

```bash
cd HPINN && pytest -q tests
cd HPINN && pytest -q tests/test_model_variants.py::test_hybrid_correction_model_exposes_components   # single test
```

`test_dataset_shapes.py` / `test_inference_api.py` require `data/de440.bsp` to be present (`find_local_kernel()` looks only in `data/`); they build a small dataset from the kernel rather than mocking it.

Paper figures — run after checkpoints exist; each script writes into its own `plots/` and skips models whose checkpoint is missing:

```bash
python comparison/paper1_stability/plot_paper1.py       # paper 1: extrapolation, energy, angular momentum
python comparison/paper2_benchmark/plot_paper2.py       # paper 2: RMSE bars, per-body heatmap, plotly 3D
python comparison/paper2_benchmark/aggiunte/plot_aggiunte_paper2.py   # paper 2 addenda + tables.md
python comparison/paper3_optimization/plot_paper3.py    # paper 3: stage histories (reads artifacts/history_*.json)
python comparison/paper4_residuals/plot_paper4.py       # paper 4: HPINN Δr correction
```

The checkpoint paths are hardcoded in each script's `CHECKPOINTS` dict and point at committed run outputs: `MLP/artifacts/emulator_stage1.pt`, `PINN/artifacts_768x8/final_checkpoint.pt/checkpoint_physics.pt` (note: `final_checkpoint.pt` there is a **directory**, because `--checkpoint-path` is a directory in multi-stage mode), `HPINN/artifacts/final_checkpoint.pt`. Update those constants when pointing the comparison at a fresh run.

## Architecture

### One model class, three configurations

`EmulatorModel` ([HPINN/src/solsys_emulator/model.py](HPINN/src/solsys_emulator/model.py)) is the single architecture; `ModelConfig` flags select the variant.

```
t (normalized scalar)
  → Fourier encoding: [t, sin(2πf·t), cos(2πf·t)], f log- or linear-spaced
  → shared backbone (plain MLP | residual blocks, SiLU)
  → optional per-body embedding concat
  → optional InteractionBlock stack (all-pairs message passing over bodies)
  → per-body heads → base output [N, B, 3 or 6]
  → if hybrid_correction: + per-body correction heads × softplus(per-body gain)
```

`forward_components()` returns `(final, base, correction)` — used by the correction-magnitude regularizer and by HPINN diagnostics. `state_mode="full"` emits 6D states directly; `state_mode="position_only"` emits 3D positions and everything else is differentiated from them.

Trained configurations: MLP = plain backbone, no embeddings/interaction/correction, position-only, data loss only. PINN = same skeleton with N-body physics losses. HPINN = residual backbone + body embeddings + interaction layers + `hybrid_correction=True`.

### Position-only + AutoDiff derivatives

All current runs use `state_mode="position_only"`: velocity and acceleration are derived from the position network by forward-mode AutoDiff rather than predicted. `_position_only_to_full_state_norm` / `_position_only_hybrid_to_full_state_norm` in `train.py` use **`jacfwd` with the vmapped functions cached on the module as `model._vmap_fns`**. This is load-bearing: the original `jacrev` + per-batch `vmap` re-tracing saturated 48 GB of VRAM and took ~60 s/epoch (full write-up in [memory.md](memory.md)). Do not reintroduce `jacrev` here, and keep the cache — the closure captures the module, so live parameter updates are still picked up.

Derivatives are computed in normalized time and rescaled by `1/time_std_seconds` (and its square) to reach km/s and km/s².

### Training loop

`train_emulator()` in `<project>/src/solsys_emulator/train.py` is a single large function handling split, dataloaders, optional DDP, the loss schedule, checkpointing, TensorBoard, and W&B. `TrainConfig` exposes each physics term as a *weight + start_epoch + warmup_epochs* triple:

- `nbody_*` — N-body acceleration residual vs `mu`-weighted Newtonian gravity, evaluated on separately sampled collocation points, with softening (`nbody_softening_km`) and a relative floor. `adaptive_nbody_balance` rescales it toward `nbody_target_fraction` of the total loss via an EMA.
- `energy_*`, `angular_momentum_*` — conservation penalties.
- `correction_*` — keeps HPINN's `Δr` small; without it the hybrid degenerates into a plain MLP.
- `physics_*` (velocity/position consistency), `smoothness_*`.

`--training-mode unified` runs one stage from `build_unified_profile()`; `--training-mode multi` runs coarse → refine → physics from `build_multi_stage_configs()`, writing `checkpoint_{coarse,refine,physics}.pt` + `history_*.json` and resuming from whichever stage checkpoints already exist. Hyperparameters are hardcoded in those two builder functions per project (CPU and GPU profiles differ) — the CLI only overrides epochs / batch size / collocation points.

Checkpoint payload: `model_state_dict`, `model_kwargs`, `scaler`, `bodies`, `time_normalization {mean, std}`, `metadata`, `train_config`. `EphemerisEmulator.from_checkpoint()` ([inference.py](HPINN/src/solsys_emulator/inference.py)) reconstructs the model from that and exposes `predict_state(iso_time)` / `predict_trajectory(...)`, warning when asked to extrapolate outside the training window.

### Data and conventions

`data/dataset_de440.npz` is shared by all three projects: `states` `(58441, 10, 6)`, 2010-01-01 → 2030-01-01 sampled every 3 h, bodies in the fixed order `sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus, neptune`. Frame ICRS barycentric, timescale TDB, units km / km/s, `mu` in km³/s². Body order is positional throughout — `mu_array()` in `constants.py`, per-body heads, and scaler axes all assume it.

States are normalized per-body-per-feature (`StateScaler`, `mode="per_body_feature"`) so inner and outer planets contribute comparably; time is standardized to zero mean / unit std.

Note `constants.py` contains a `LEOPARDD_DICT` shim and a `get_mu()` fallback that *warns and returns 1.0* for unknown bodies instead of raising — a silent-wrong-physics trap if a body name is ever misspelled.

### Logging

W&B is on by default (`use_wandb=True`) with project names `mlp-/pinn-/hpinn-solar-system`; run `wandb login` first or set `WANDB_MODE=offline`. TensorBoard events go to `logs/runs/`. Both only emit from rank 0.

DDP support lives inside `train.py` (`TrainConfig.distributed`, `DistributedSampler`, `DDP` wrapper). `build_unified_profile()` sets `distributed=True` automatically when `torch.cuda.device_count() > 1`, and `train_emulator()` then **raises** unless launched under `torchrun` — relevant only on multi-GPU hosts (this DGX has one GB10). The `train_ddp.py` and `monitor_training.py` referenced in `PINN/docs/` and `HPINN/docs/` do not exist; treat those docs as aspirational.
