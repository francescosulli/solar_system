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

#SBATCH --job-name=pinn-solar-system
#SBATCH -A astreo 
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=giovanni.billo@studenti.units.it,francesco.sulli@studenti.units.it

# =============================================================================
# Paths & run directory
# =============================================================================

mkdir -p logs
JOB_NAME="${SLURM_JOB_NAME:-local_run}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="checkpoints/${JOB_NAME}" 
PLOTS_DIR="plots/${JOB_NAME}" 
RUN_DIR="logs/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${PLOTS_DIR}"

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
# Environment setup
# =============================================================================


UV_BIN="$HOME/dlprojenv/bin/uv" # change accordingly
VENV_DIR=".venv"

source "$VENV_DIR"/bin/activate

"$UV_BIN" sync 

# Limit NumPy/OpenBLAS threads to the reserved CPU count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Use only the GPU assigned by SLURM
# export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

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

"$VENV_DIR"/bin/python verify_setup.py 

# "$VENV_DIR"/bin/python solsys_setup.py \
# 	--dataset-path "$(pwd)/data/leopardd_dataset.npz" \
# 	--dataset-type "tle" 

"$VENV_DIR"/bin/python do_setup.py \
	--dataset-path "$(pwd)/data/dataset_demo.npz" \
	--dataset-type "jpl" 

# "$VENV_DIR"/bin/python train.py --device cpu --epochs 10 --training-mode=multi 
## when training on a single GPU, don't spawn multiple processes: it will crash

"$VENV_DIR/bin/torchrun" --nproc_per_node=1 train.py \
  --training-mode multi \
  --project-name "pinn-solar-system" \
  --dataset-path "$(pwd)/data/dataset_demo.npz" \
  --checkpoint-path "$CHECKPOINT_DIR" \
  --plots-dir "$PLOTS_DIR"

#   --training-mode unified \
#   --project-name "LEOPARDD" \
#   --dataset-path "$(pwd)/data/fengyun_dataset.npz" \
#

EXIT_CODE=$?

echo ""
echo "============================================================"
echo " End time    : $(date)"
echo " Exit code   : $EXIT_CODE"
echo " Run dir     : ${RUN_DIR}"
echo "============================================================"

exit $EXIT_CODE
