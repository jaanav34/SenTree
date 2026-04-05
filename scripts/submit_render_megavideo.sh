#!/usr/bin/env bash
#
# Submit the mega-video render as a Slurm job (with optional account passthrough).
#
# Usage:
#   bash scripts/submit_render_megavideo.sh
#
# Optional env vars:
#   SENTREE_SLURM_ACCOUNT=<account>        (recommended on Gautschi)
#   SENTREE_SBATCH_EXTRA_ARGS="--qos=..."  (optional passthrough)
#   SENTREE_MEGA_MODE=grid|cycle
#   SENTREE_MEGA_GRID_KIND=delta|intervention
#   SENTREE_MEGA_NCOLS=6
#   SENTREE_MEGA_OUT=outputs/videos/interventions_grid_years.mp4
#   SENTREE_MEGA_ANIMATE_YEARS=1
#   SENTREE_RENDER_FPS=4
#   SENTREE_MEGA_PROGRESS_EVERY=1
#   SENTREE_SERIES_DIR=outputs/roi/risk_series
#
set -euo pipefail

SLURM_ACCOUNT="${SENTREE_SLURM_ACCOUNT:-}"
SBATCH_EXTRA_ARGS="${SENTREE_SBATCH_EXTRA_ARGS:-}"

mkdir -p logs

set +e
CMD=(sbatch)
if [[ -n "$SLURM_ACCOUNT" ]]; then
  CMD+=(--account "$SLURM_ACCOUNT")
fi
if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  CMD+=($SBATCH_EXTRA_ARGS)
fi
CMD+=(--export=ALL jobs/render_megavideo.sbatch)

"${CMD[@]}"
RC=$?
set -e

if [[ "$RC" -ne 0 ]]; then
  echo ""
  echo "sbatch failed."
  echo "If you're on Gautschi, you likely need to specify an account:"
  echo "  export SENTREE_SLURM_ACCOUNT=<your_account>"
  echo "  bash scripts/submit_render_megavideo.sh"
  echo "Or submit manually:"
  echo "  sbatch -A <your_account> jobs/render_megavideo.sbatch"
  exit "$RC"
fi

