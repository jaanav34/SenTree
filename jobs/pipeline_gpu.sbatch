#!/bin/bash
#SBATCH -A mlp
#SBATCH -p smallgpu
#SBATCH -J sentree_gpu
#SBATCH -o logs/pipeline_gpu.%j.out
#SBATCH -e logs/pipeline_gpu.%j.err
#SBATCH -t 04:00:00
#SBATCH -c 64
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --force purge
module load modtree/gpu
module load cuda/12.6.0
module load gcc/14.1.0
module load python/3.11.9

source .venv/bin/activate

python -u -c "import torch; print('cuda available:', torch.cuda.is_available()); print('cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -u scripts/run_pipeline.py
