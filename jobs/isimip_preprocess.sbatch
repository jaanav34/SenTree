#!/bin/bash
#SBATCH -A mlp
#SBATCH -p cpu
#SBATCH -J isimip_prep
#SBATCH -o logs/isimip_prep.%j.out
#SBATCH -e logs/isimip_prep.%j.err
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=32G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --force purge
module load modtree/cpu
module load gcc/14.1.0
module load python/3.11.9

if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "../.venv" ]; then
  source ../.venv/bin/activate
else
  echo "ERROR: no .venv found in repo or parent dir" >&2
  exit 1
fi

python -u -c "
from src.data.load_isimip import load_climate_data
d = load_climate_data(data_dir='data/processed', raw_dir='../data/raw')
print('DONE')
print('tas', d['tas'].shape, 'pr', d['pr'].shape)
print('years', int(d['years'][0]), 'to', int(d['years'][-1]), 'n=', len(d['years']))
"
