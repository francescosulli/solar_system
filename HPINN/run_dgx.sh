#!/bin/bash
# =============================================================================
# run_dgx.sh — HPINN Solar System Emulator (ibrido fisica + correzione appresa)
# Target: DGX Spark (NVIDIA GB10, CUDA 13.0, aarch64, single GPU)
#
# Uso:
#   bash run_dgx.sh [unified|multi]
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv"

# Esegui tutto da dentro la cartella del modello
cd "$SCRIPT_DIR"

TRAINING_MODE="${1:-unified}"
PROJECT_NAME="hpinn-solar-system"
DATASET_PATH="$REPO_ROOT/data/dataset_de440.npz"  # dataset condiviso

RUN_ID="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/${RUN_ID}"
PLOTS_DIR="plots/${RUN_ID}"
RUN_LOG_DIR="logs/runs/${RUN_ID}"

mkdir -p "$CHECKPOINT_DIR" "$PLOTS_DIR" "$RUN_LOG_DIR" logs

exec > >(tee -a "${RUN_LOG_DIR}/run.log") 2>&1

echo "============================================================"
echo " Script     : HPINN run_dgx.sh"
echo " Mode       : $TRAINING_MODE"
echo " Run ID     : $RUN_ID"
echo " Start time : $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Attiva venv
# ---------------------------------------------------------------------------
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv non trovato in $VENV"
    echo "Esegui prima il setup: bash $REPO_ROOT/PINN/run_dgx.sh"
    exit 1
fi
source "$VENV/bin/activate"
echo "Venv attivo: $(which python)"

# ---------------------------------------------------------------------------
# 2. Installa il package in editable mode se necessario
# ---------------------------------------------------------------------------
if ! python -c "import solsys_emulator" 2>/dev/null; then
    echo "Installo solsys_emulator in editable mode..."
    pip install -e . -q
fi

# ---------------------------------------------------------------------------
# 3. Variabili d'ambiente
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# 4. GPU snapshot
# ---------------------------------------------------------------------------
echo ""
echo "=== GPU Info ==="
nvidia-smi | tee "${RUN_LOG_DIR}/nvidia-smi.start.txt"

python - <<'EOF'
import torch
print(f"torch        : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPU          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
EOF

# ---------------------------------------------------------------------------
# 5. Dataset — usa quello generato da PINN
# ---------------------------------------------------------------------------
if [ ! -f "$DATASET_PATH" ]; then
    echo "Dataset non trovato in $DATASET_PATH"
    echo "Genera prima il dataset eseguendo: bash $REPO_ROOT/PINN/run_dgx.sh"
    exit 1
fi
echo "Dataset: $DATASET_PATH"

# ---------------------------------------------------------------------------
# 6. Training HPINN
# ---------------------------------------------------------------------------
echo ""
echo "--- Training HPINN (mode: $TRAINING_MODE) ---"

python train.py \
    --training-mode "$TRAINING_MODE" \
    --device cuda \
    --project-name "$PROJECT_NAME" \
    --dataset-path "$DATASET_PATH" \
    --checkpoint-path "$CHECKPOINT_DIR" \
    --plots-dir "$PLOTS_DIR"

EXIT_CODE=$?

nvidia-smi | tee "${RUN_LOG_DIR}/nvidia-smi.end.txt" || true

echo ""
echo "============================================================"
echo " End time   : $(date)"
echo " Exit code  : $EXIT_CODE"
echo " Checkpoint : $CHECKPOINT_DIR"
echo " Plots      : $PLOTS_DIR"
echo "============================================================"

exit $EXIT_CODE
