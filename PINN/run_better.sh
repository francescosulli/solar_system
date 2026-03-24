#!/bin/bash
# =============================================================================
# SLURM job script — Solar System PINN Emulator
#
# Hardware target : 1× NVIDIA V100 (32 GB), GPU partition
# Wall time       : 2 hours
# Memory          : 64 GB RAM
# CPUs            : 8
#
# Core-count rationale
# --------------------
# The training loop is entirely GPU-bound (PyTorch + CUDA).  CPU cores are
# needed only for:
#   • 1  main Python / orchestration thread
#   • 4  DataLoader prefetch workers (num_workers=4 keeps the GPU fed without
#        wasting idle cores — TensorDataset is in-memory so more workers give
#        no benefit)
#   • 2  spare cores for Astropy/NumPy BLAS calls during dataset build and
#        inference (sample_states iterates over 10 bodies serially, but NumPy
#        internal BLAS/OpenBLAS can use a couple of threads)
#   → Total: 8 CPUs  (requesting more would sit idle and waste allocation)
# =============================================================================

#SBATCH --job-name=solsys_pinn
#SBATCH -A dssc
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=giovanni.billo@studenti.units.it,francesco.sulli@studenti.units.it

# =============================================================================
# Experiment parameters (edit here before submitting)
# =============================================================================

START_YEAR=${START_YEAR:-2010}
END_YEAR=${END_YEAR:-2030}
STEP_HOURS=${STEP_HOURS:-3.0}
EPOCHS_STAGE1=${EPOCHS_STAGE1:-1000}
EPOCHS_STAGE2=${EPOCHS_STAGE2:-350}
BATCH_SIZE=${BATCH_SIZE:-384}
HIDDEN_DIM=${HIDDEN_DIM:-384}
NUM_LAYERS=${NUM_LAYERS:-6}
FOURIER_FEATURES=${FOURIER_FEATURES:-24}
INFER_START=${INFER_START:-"2025-01-01T00:00:00"}
INFER_END=${INFER_END:-"2026-01-01T00:00:00"}
INFER_STEP_HOURS=${INFER_STEP_HOURS:-6.0}
KERNEL=${KERNEL:-"PINN/data/de440.bsp"}
OUTPUT_DIR=${OUTPUT_DIR:-"artifacts"}
DEVICE=${DEVICE:-"cuda"}

echo "=== Experiment parameters ==="
echo "  Start year       : ${START_YEAR}"
echo "  End year         : ${END_YEAR}"
echo "  Step hours       : ${STEP_HOURS}"
echo "  Epochs stage 1   : ${EPOCHS_STAGE1}"
echo "  Epochs stage 2   : ${EPOCHS_STAGE2}"
echo "  Batch size       : ${BATCH_SIZE}"
echo "  Hidden dim       : ${HIDDEN_DIM}"
echo "  Num layers       : ${NUM_LAYERS}"
echo "  Fourier features : ${FOURIER_FEATURES}"
echo "  Infer start      : ${INFER_START}"
echo "  Infer end        : ${INFER_END}"
echo "  Infer step hours : ${INFER_STEP_HOURS}"
echo "  Kernel           : ${KERNEL}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "  Device           : ${DEVICE}"
echo "============================="

# =============================================================================
# Paths & run directory
# =============================================================================

mkdir -p logs

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="logs/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

# Mirror all stdout/stderr into the run folder as well
exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

echo "============================================================"
echo " Job name    : $SLURM_JOB_NAME"
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $SLURMD_NODENAME"
echo " CPUs/task   : $SLURM_CPUS_PER_TASK"
echo " Start time  : $(date)"
echo " Run dir     : ${RUN_DIR}"
echo "============================================================"

# =============================================================================
# Save run metadata
# =============================================================================

cat > "${RUN_DIR}/meta.txt" << EOF
time:             $(date -Is)
job_id:           ${SLURM_JOB_ID}
job_name:         ${SLURM_JOB_NAME}
node:             ${SLURMD_NODENAME}
cpus_per_task:    ${SLURM_CPUS_PER_TASK}
start_year:       ${START_YEAR}
end_year:         ${END_YEAR}
step_hours:       ${STEP_HOURS}
epochs_stage1:    ${EPOCHS_STAGE1}
epochs_stage2:    ${EPOCHS_STAGE2}
batch_size:       ${BATCH_SIZE}
hidden_dim:       ${HIDDEN_DIM}
num_layers:       ${NUM_LAYERS}
fourier_features: ${FOURIER_FEATURES}
infer_start:      ${INFER_START}
infer_end:        ${INFER_END}
infer_step_hours: ${INFER_STEP_HOURS}
kernel:           ${KERNEL}
output_dir:       ${OUTPUT_DIR}
device:           ${DEVICE}
EOF

echo "Metadata saved to ${RUN_DIR}/meta.txt"

# =============================================================================
# Environment setup
# =============================================================================

module purge
module load cuda/12.8

source .venv/bin/activate

uv sync --active

# Limit NumPy/OpenBLAS threads to the reserved CPU count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Use only the GPU assigned by SLURM
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# =============================================================================
# GPU logging setup
# =============================================================================

echo ""
echo "=== NVIDIA-SMI debug setup ==="

# One-shot snapshot at start
nvidia-smi | tee "${RUN_DIR}/nvidia-smi.start.txt"

# Structured CSV snapshot at start
nvidia-smi --query-gpu=timestamp,index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit \
    --format=csv,noheader,nounits \
    | tee "${RUN_DIR}/nvidia-smi.start.csv"

# Compute processes at start
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits \
    | tee "${RUN_DIR}/nvidia-smi.processes.start.csv" || true

NVIDIA_SMI_PID=""

cleanup() {
    echo ""
    echo "=== Cleanup ==="
    if [[ -n "${NVIDIA_SMI_PID}" ]] && kill -0 "${NVIDIA_SMI_PID}" 2>/dev/null; then
        echo "Stopping nvidia-smi sampler (PID ${NVIDIA_SMI_PID})"
        kill "${NVIDIA_SMI_PID}" 2>/dev/null || true
        wait "${NVIDIA_SMI_PID}" 2>/dev/null || true
    fi
    # Final snapshot at exit
    nvidia-smi | tee "${RUN_DIR}/nvidia-smi.end.txt" || true
    nvidia-smi --query-gpu=timestamp,index,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
        --format=csv,noheader,nounits \
        | tee "${RUN_DIR}/nvidia-smi.end.csv" || true
}
trap cleanup EXIT INT TERM

# Background GPU sampling at 1 Hz
(
    echo "timestamp,index,memory.used(MiB),util.gpu(%),util.mem(%),temp(C),power(W)"
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
        --format=csv,noheader,nounits -l 1
) > "${RUN_DIR}/nvidia-smi.csv" &
NVIDIA_SMI_PID=$!

# Process-level sampling (best-effort)
if command -v nvidia-smi >/dev/null 2>&1; then
    (
        echo "pmon_snapshot(pid,sm,mem,enc,dec,command) - sampled every 1s"
        nvidia-smi pmon -s um -o DT -d 1 2>/dev/null || true
    ) > "${RUN_DIR}/nvidia-smi.pmon.csv" &
fi

echo "nvidia-smi sampler PID : ${NVIDIA_SMI_PID}"
echo "GPU timeseries log     : ${RUN_DIR}/nvidia-smi.csv"
echo "Process monitor log    : ${RUN_DIR}/nvidia-smi.pmon.csv"
echo "============================="

# =============================================================================
# Sanity checks
# =============================================================================

echo ""
echo "--- Python / torch ---"
echo "Python : $(which python)"
python --version
python -c "
import torch
print('torch        :', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU          :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"

# =============================================================================
# Run the pipeline
# =============================================================================

echo ""
echo "--- Starting pipeline ---"

# python run_pipeline.py \
#     --start              "${START_YEAR}-01-01T00:00:00" \
#     --end                "${END_YEAR}-01-01T00:00:00" \
#     --step-hours         "${STEP_HOURS}" \
#     --epochs-stage1      "${EPOCHS_STAGE1}" \
#     --epochs-stage2      "${EPOCHS_STAGE2}" \
#     --batch-size         "${BATCH_SIZE}" \
#     --hidden-dim         "${HIDDEN_DIM}" \
#     --num-layers         "${NUM_LAYERS}" \
#     --fourier-features   "${FOURIER_FEATURES}" \
#     --infer-start        "${INFER_START}" \
#     --infer-end          "${INFER_END}" \
#     --infer-step-hours   "${INFER_STEP_HOURS}" \
#     --output-dir         "${OUTPUT_DIR}" \
#     --device             "${DEVICE}" \
#     --kernel             "${KERNEL}"
#     # --do-stage2        # uncomment to enable physics fine-tuning (Stage 2)
#     # --skip-dataset     # uncomment to reuse existing data/dataset_demo.npz
#     # --skip-train       # uncomment to skip training, run inference only
python solsys_setup.py \
	--dataset-path $(pwd)/data/dataset_demo.npz \

python PINN_train.py \
	--dataset-path $(pwd)/data/dataset_demo.npz \
	--checkpoint-path $(pwd)/artifacts \
	--plots-dir $(pwd)/plots/ 

EXIT_CODE=$?

echo ""
echo "============================================================"
echo " End time    : $(date)"
echo " Exit code   : $EXIT_CODE"
echo " Run dir     : ${RUN_DIR}"
echo "============================================================"

exit $EXIT_CODE
