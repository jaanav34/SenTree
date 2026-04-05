#!/bin/bash
#SBATCH -A mlp
#SBATCH -p cpu
#SBATCH -J sentree_pipe
#SBATCH -o logs/pipeline.%j.out
#SBATCH -e logs/pipeline.%j.err
#SBATCH -t 04:00:00
#SBATCH -c 16
#SBATCH --mem=64G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --force purge
module load modtree/cpu
module load gcc/14.1.0
module load python/3.11.9
source .venv/bin/activate

python -u scripts/run_pipeline.py
