#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
EXPORT_SCRIPT="$ROOT_DIR/scripts/export_gnn_playback_data.py"
PLAYBACK_SOURCE="$ROOT_DIR/outputs/roi/gnn_training_history.npz"
APP_DIR="$ROOT_DIR/apps/gnn-playback"
MAX_NODES="${SENTREE_PLAYBACK_MAX_NODES:-5000}"
MAX_EDGES="${SENTREE_PLAYBACK_MAX_EDGES:-200000}"
SEED="${SENTREE_PLAYBACK_SEED:-0}"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: Expected virtualenv python at $VENV_PY" 1>&2
  echo "Run ./setup.sh first." 1>&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm was not found on PATH." 1>&2
  echo "Install Node.js, then re-run this script." 1>&2
  exit 1
fi

if [ ! -f "$PLAYBACK_SOURCE" ]; then
  echo "ERROR: Missing $PLAYBACK_SOURCE" 1>&2
  echo "Run .venv/bin/python scripts/run_pipeline.py first to generate GNN training history." 1>&2
  exit 1
fi

cd "$ROOT_DIR"
"$VENV_PY" "$EXPORT_SCRIPT" --max-nodes "$MAX_NODES" --max-edges "$MAX_EDGES" --seed "$SEED"

cd "$APP_DIR"
if [ ! -d node_modules ]; then
  echo "Installing React app dependencies..."
  npm install
fi

exec npm run dev
