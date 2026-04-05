#!/usr/bin/env bash
#
# Submit the comparison-video render Slurm array cleanly.
#
# Usage:
#   bash scripts/submit_render_comparisons.sh
#
# Optional env vars:
#   SENTREE_KEYS_FILE=outputs/roi/risk_series/intervention_keys.txt
#   SENTREE_RENDER_FPS=2
#   SENTREE_RENDER_SCALE_FACTOR=4
#   SENTREE_RENDER_DPI=80
#   SENTREE_FFMPEG_THREADS=$SLURM_CPUS_PER_TASK
#
set -euo pipefail

KEYS_FILE="${SENTREE_KEYS_FILE:-outputs/roi/risk_series/intervention_keys.txt}"

if [[ ! -f "$KEYS_FILE" ]]; then
  echo "Missing keys file: $KEYS_FILE"
  echo "Generating keys file..."
  python scripts/export_intervention_keys.py --out-dir "$(dirname "$KEYS_FILE")"
fi

N="$(wc -l < "$KEYS_FILE" | tr -d ' ')"
if [[ -z "$N" || "$N" -le 0 ]]; then
  echo "No keys found in $KEYS_FILE"
  exit 1
fi

mkdir -p logs
echo "Submitting render array with N=$N tasks from $KEYS_FILE"

sbatch \
  --export=ALL,SENTREE_KEYS_FILE="$KEYS_FILE" \
  --array="1-$N" \
  jobs/render_comparisons_array.sbatch
