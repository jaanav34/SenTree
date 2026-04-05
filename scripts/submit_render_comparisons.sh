#!/usr/bin/env bash
#
# Submit the comparison-video render Slurm array cleanly.
#
# Usage:
#   bash scripts/submit_render_comparisons.sh
#
# Optional env vars:
#   SENTREE_SERIES_DIR=outputs/roi/risk_series
#   SENTREE_KEYS_FILE=scripts/.cache/intervention_keys.txt
#   SENTREE_NAMES_JSON=scripts/.cache/intervention_names.json
#   SENTREE_RENDER_FPS=2
#   SENTREE_RENDER_SCALE_FACTOR=4
#   SENTREE_RENDER_DPI=80
#   SENTREE_FFMPEG_THREADS=$SLURM_CPUS_PER_TASK
#
set -euo pipefail

SERIES_DIR="${SENTREE_SERIES_DIR:-outputs/roi/risk_series}"
KEYS_FILE="${SENTREE_KEYS_FILE:-scripts/.cache/intervention_keys.txt}"
NAMES_JSON="${SENTREE_NAMES_JSON:-scripts/.cache/intervention_names.json}"

mkdir -p "$(dirname "$KEYS_FILE")"

if [[ ! -f "$KEYS_FILE" ]]; then
  echo "Missing keys file: $KEYS_FILE"
  echo "Generating keys/names manifest..."
  python scripts/export_intervention_keys.py --out-dir "$(dirname "$KEYS_FILE")" --series-dir "$SERIES_DIR"
fi

N="$(wc -l < "$KEYS_FILE" | tr -d ' ')"
if [[ -z "$N" || "$N" -le 0 ]]; then
  echo "No keys found in $KEYS_FILE"
  exit 1
fi

mkdir -p logs
echo "Submitting render array with N=$N tasks from $KEYS_FILE"

sbatch \
  --export=ALL,SENTREE_KEYS_FILE="$KEYS_FILE",SENTREE_NAMES_JSON="$NAMES_JSON" \
  --array="1-$N" \
  jobs/render_comparisons_array.sbatch
