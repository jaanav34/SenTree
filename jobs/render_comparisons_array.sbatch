#!/usr/bin/env bash
#
# Render comparison videos for all interventions via a Slurm job array.
# This assumes the pipeline was run with:
#   SENTREE_SAVE_RISK_SERIES_NPZ=1
# and created:
#   outputs/roi/risk_series/baseline.npz
#   outputs/roi/risk_series/intervention_<key>.npz
#
# A keys/names manifest is required for Slurm arrays; the wrapper generates it
# under `scripts/.cache/` by default and exports paths via env vars:
#   SENTREE_KEYS_FILE, SENTREE_NAMES_JSON
#
# Submit via the wrapper:
#   scripts/submit_render_comparisons.sh
#
# Or manually:
#   export SENTREE_KEYS_FILE=scripts/.cache/intervention_keys.txt
#   N=$(wc -l < "$SENTREE_KEYS_FILE")
#   sbatch --export=ALL --array=1-$N jobs/render_comparisons_array.sbatch
#

#SBATCH -J sentree_render_cmp
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o logs/render_cmp.%A_%a.out
#SBATCH -e logs/render_cmp.%A_%a.err

set -euo pipefail

KEYS_FILE="${SENTREE_KEYS_FILE:-outputs/roi/risk_series/intervention_keys.txt}"
NAMES_JSON="${SENTREE_NAMES_JSON:-outputs/roi/risk_series/intervention_names.json}"

if [[ ! -f "$KEYS_FILE" ]]; then
  echo "Missing KEYS_FILE: $KEYS_FILE"
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$TASK_ID" ]]; then
  echo "SLURM_ARRAY_TASK_ID not set (did you submit as an array?)"
  exit 1
fi

KEY="$(sed -n "${TASK_ID}p" "$KEYS_FILE" | tr -d '\r')"
if [[ -z "$KEY" ]]; then
  echo "No key found for task id $TASK_ID in $KEYS_FILE"
  exit 1
fi

echo "Rendering comparison video for key=$KEY (task $TASK_ID)"
echo "FPS=${SENTREE_RENDER_FPS:-4} SCALE=${SENTREE_RENDER_SCALE_FACTOR:-8} DPI=${SENTREE_RENDER_DPI:-}"

if [[ -z "${SENTREE_FFMPEG_THREADS:-}" ]]; then
  export SENTREE_FFMPEG_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
echo "FFMPEG_THREADS=${SENTREE_FFMPEG_THREADS} PRESET=${SENTREE_FFMPEG_PRESET:-} CRF=${SENTREE_FFMPEG_CRF:-}"

python scripts/render_comparison_from_npz.py --key "$KEY" --names-json "$NAMES_JSON"
