#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
APP_PATH="$ROOT_DIR/src/dashboard/app.py"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: Expected virtualenv python at $VENV_PY" 1>&2
  echo "Run ./setup.sh first." 1>&2
  exit 1
fi

cd "$ROOT_DIR"
exec "$VENV_PY" -m streamlit run "$APP_PATH"
