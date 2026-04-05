#!/usr/bin/env bash
#
# SenTree pipeline (GPU allocation, CPU-threaded where possible).
#
# Submit (Gautschi example):
#   export SENTREE_SLURM_ACCOUNT=mlp
#   sbatch -A "$SENTREE_SLURM_ACCOUNT" jobs/pipeline_ai_gpu.sbatch
#
# Notes:
# - Many steps are CPU-bound (NumPy/BLAS, Köppen–Geiger, Matplotlib/ffmpeg).
# - `#SBATCH -c` controls how many CPU cores you get; the exports below make sure
#   libraries and encoders actually use them.
#

#SBATCH -J sentree_ai
#SBATCH -p ai
#SBATCH -o logs/pipeline_ai.%j.out
#SBATCH -e logs/pipeline_ai.%j.err
#SBATCH -t 04:00:00
#SBATCH -c 14
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# Threading: Slurm sets SLURM_CPUS_PER_TASK inside the job allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"

# Project knobs that can use CPU parallelism when enabled.
export SENTREE_KG_WORKERS="${SENTREE_KG_WORKERS:-${SLURM_CPUS_PER_TASK}}"
export SENTREE_FFMPEG_THREADS="${SENTREE_FFMPEG_THREADS:-${SLURM_CPUS_PER_TASK}}"

module --force purge
module load modtree/gpu
module load gcc/14.1.0
module load cuda/12.6.1

source .venv/bin/activate

echo "=== MODULE LIST ==="
module -t list || true
echo "=== CPU THREADS ==="
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "SENTREE_KG_WORKERS=${SENTREE_KG_WORKERS} SENTREE_FFMPEG_THREADS=${SENTREE_FFMPEG_THREADS}"

echo "=== GPU ==="
nvidia-smi || true
echo "=== TORCH ==="
python -u -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== RUN PIPELINE ==="
python -u scripts/run_pipeline.py

