#!/usr/bin/env bash
#
# Render a mega (grid) intervention video from saved NPZ risk series.
#
# This expects the pipeline has produced:
#   outputs/roi/risk_series/baseline.npz
#   outputs/roi/risk_series/intervention_<key>.npz  (for each key)
#
# Submit:
#   sbatch -A <account> jobs/render_megavideo.sbatch
# or:
#   export SENTREE_SLURM_ACCOUNT=<account>; bash scripts/submit_render_megavideo.sh
#
# Tail logs:
#   tail -f logs/megavideo.<jobid>.out
#
# Optional env vars:
#   SENTREE_MEGA_MODE=grid|cycle
#   SENTREE_MEGA_GRID_KIND=delta|intervention
#   SENTREE_MEGA_NCOLS=6
#   SENTREE_MEGA_OUT=outputs/videos/interventions_grid_years.mp4
#   SENTREE_MEGA_ANIMATE_YEARS=1          (default 1)
#   SENTREE_RENDER_FPS=4
#   SENTREE_MEGA_PROGRESS_EVERY=1
#   SENTREE_SERIES_DIR=outputs/roi/risk_series
#

#SBATCH -J sentree_megavideo
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH -o logs/megavideo.%j.out
#SBATCH -e logs/megavideo.%j.err

set -euo pipefail

MODE="${SENTREE_MEGA_MODE:-grid}"
NCOLS="${SENTREE_MEGA_NCOLS:-6}"
OUT="${SENTREE_MEGA_OUT:-outputs/videos/interventions_grid_years.mp4}"
ANIMATE="${SENTREE_MEGA_ANIMATE_YEARS:-1}"
SERIES_DIR="${SENTREE_SERIES_DIR:-outputs/roi/risk_series}"
PROG_EVERY="${SENTREE_MEGA_PROGRESS_EVERY:-1}"
GRID_KIND="${SENTREE_MEGA_GRID_KIND:-delta}"

echo "Megavideo job starting..."
echo "MODE=$MODE GRID_KIND=$GRID_KIND NCOLS=$NCOLS OUT=$OUT ANIMATE=$ANIMATE SERIES_DIR=$SERIES_DIR"
echo "FPS=${SENTREE_RENDER_FPS:-4} SCALE=${SENTREE_RENDER_SCALE_FACTOR:-} DPI=${SENTREE_RENDER_DPI:-}"

ARGS=(--mode "$MODE" --grid-kind "$GRID_KIND" --ncols "$NCOLS" --out "$OUT" --series-dir "$SERIES_DIR" --progress --progress-every "$PROG_EVERY")
if [[ "$ANIMATE" == "1" || "$ANIMATE" == "true" || "$ANIMATE" == "TRUE" ]]; then
  ARGS+=(--animate-years)
fi

python scripts/render_megavideo_from_npz.py "${ARGS[@]}"
echo "Megavideo job done."
