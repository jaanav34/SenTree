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
#   SENTREE_SLURM_ACCOUNT=<account>        (recommended on Gautschi)
#   SENTREE_SBATCH_EXTRA_ARGS="--qos=..."  (optional passthrough)
#   SENTREE_RENDER_FPS=2
#   SENTREE_RENDER_SCALE_FACTOR=4
#   SENTREE_RENDER_DPI=80
#   SENTREE_FFMPEG_THREADS=$SLURM_CPUS_PER_TASK
#
set -euo pipefail

SERIES_DIR="${SENTREE_SERIES_DIR:-outputs/roi/risk_series}"
KEYS_FILE="${SENTREE_KEYS_FILE:-scripts/.cache/intervention_keys.txt}"
NAMES_JSON="${SENTREE_NAMES_JSON:-scripts/.cache/intervention_names.json}"
SLURM_ACCOUNT="${SENTREE_SLURM_ACCOUNT:-}"
SBATCH_EXTRA_ARGS="${SENTREE_SBATCH_EXTRA_ARGS:-}"

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

set +e
CMD=(sbatch)
if [[ -n "$SLURM_ACCOUNT" ]]; then
  CMD+=(--account "$SLURM_ACCOUNT")
fi
if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  CMD+=($SBATCH_EXTRA_ARGS)
fi
CMD+=(--export=ALL,SENTREE_KEYS_FILE="$KEYS_FILE",SENTREE_NAMES_JSON="$NAMES_JSON" --array="1-$N" jobs/render_comparisons_array.sbatch)

"${CMD[@]}"
RC=$?
set -e

if [[ "$RC" -ne 0 ]]; then
  echo ""
  echo "sbatch failed."
  echo "If you're on Gautschi, you likely need to specify an account:"
  echo "  export SENTREE_SLURM_ACCOUNT=<your_account>"
  echo "  bash scripts/submit_render_comparisons.sh"
  echo "Or submit manually:"
  echo "  sbatch -A <your_account> --array=1-$N jobs/render_comparisons_array.sbatch"
  exit "$RC"
fi
