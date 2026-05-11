#!/bin/bash
# =============================================================================
# run_dgx.sh — PINN Solar System Emulator
# Target: DGX Spark (NVIDIA GB10, CUDA 13.0, aarch64, single GPU)
#
# Uso:
#   bash run_dgx.sh [unified|multi]
#
# Passi:
#   1. Attiva il venv condiviso (root della repo)
#   2. Installa il package in modalità editable se necessario
#   3. Scarica il kernel DE440 se assente
#   4. Genera il dataset da DE440
#   5. Lancia il training
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

TRAINING_MODE="${1:-unified}"           # unified (default) | multi
PROJECT_NAME="pinn-solar-system"
DATASET_PATH="$REPO_ROOT/data/dataset_de440.npz"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/${RUN_ID}"
PLOTS_DIR="plots/${RUN_ID}"
RUN_LOG_DIR="logs/runs/${RUN_ID}"

mkdir -p "$CHECKPOINT_DIR" "$PLOTS_DIR" "$RUN_LOG_DIR" logs

# Mirror stdout/stderr nel log di run
exec > >(tee -a "${RUN_LOG_DIR}/run.log") 2>&1

echo "============================================================"
echo " Script     : PINN run_dgx.sh"
echo " Mode       : $TRAINING_MODE"
echo " Run ID     : $RUN_ID"
echo " Start time : $(date)"
echo " Run dir    : ${RUN_LOG_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Attiva venv
# ---------------------------------------------------------------------------
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv non trovato in $VENV"
    echo "Crea il venv con: python3 -m venv $VENV"
    echo "Poi installa le dipendenze: pip install -r $REPO_ROOT/requirements-dgx.txt --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple"
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
# 3. Variabili d'ambiente performance
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_LAUNCH_BLOCKING rallenta il training — usare solo per debug
# export CUDA_LAUNCH_BLOCKING=1

# ---------------------------------------------------------------------------
# 4. GPU snapshot iniziale
# ---------------------------------------------------------------------------
echo ""
echo "=== GPU Info ==="
nvidia-smi | tee "${RUN_LOG_DIR}/nvidia-smi.start.txt"
echo "================"

# Sanity check Python/Torch
python - <<'EOF'
import torch
print(f"torch        : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPU          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"CUDA version : {torch.version.cuda}")
EOF

# ---------------------------------------------------------------------------
# 5. Verifica setup (importa tutti i moduli del package)
# ---------------------------------------------------------------------------
echo ""
echo "--- verify_setup ---"
python verify_setup.py

# ---------------------------------------------------------------------------
# 6. Genera dataset DE440 (skip se già presente)
# ---------------------------------------------------------------------------
echo ""
echo "--- Dataset setup ---"
python do_setup.py \
    --dataset-path "$DATASET_PATH" \
    --dataset-type "jpl"

# ---------------------------------------------------------------------------
# 7. Training
# ---------------------------------------------------------------------------
echo ""
echo "--- Training (mode: $TRAINING_MODE) ---"

# Singola GPU: usare python direttamente (non torchrun, non necessario)
python train.py \
    --training-mode "$TRAINING_MODE" \
    --device cuda \
    --project-name "$PROJECT_NAME" \
    --dataset-path "$DATASET_PATH" \
    --checkpoint-path "$CHECKPOINT_DIR" \
    --plots-dir "$PLOTS_DIR"

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# 8. GPU snapshot finale
# ---------------------------------------------------------------------------
nvidia-smi | tee "${RUN_LOG_DIR}/nvidia-smi.end.txt" || true

echo ""
echo "============================================================"
echo " End time   : $(date)"
echo " Exit code  : $EXIT_CODE"
echo " Checkpoint : $CHECKPOINT_DIR"
echo " Plots      : $PLOTS_DIR"
echo " Log        : ${RUN_LOG_DIR}/run.log"
echo "============================================================"

exit $EXIT_CODE
