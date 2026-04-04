#!/bin/bash
set -e

echo "=== SenTree Setup ==="

PYTHON="${PYTHON:-python}"
PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJ="${PY_VER%.*}"
PY_MIN="${PY_VER#*.}"
if [ "$PY_MAJ" = "3" ] && [ "$PY_MIN" -ge 14 ]; then
  if [ "${SENTREE_SKIP_PY_VERSION_CHECK:-}" = "1" ]; then
    echo "WARNING: Skipping Python version check (Python $PY_VER)." 1>&2
  else
  echo "ERROR: Python $PY_VER is too new for reliable wheels (numpy/torch often missing)." 1>&2
  echo "Install Python 3.12 or 3.13 and re-run, or set PYTHON=python3.12" 1>&2
  exit 1
  fi
fi

if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi

VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY" 1>&2
  exit 1
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install --require-virtualenv -r requirements.txt

# System dependency: ffmpeg (required to render MP4s)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found on PATH. MP4 rendering will fail until you install it." 1>&2
fi

# PyG CPU install (safe for hackathon)
"$VENV_PY" -m pip install --require-virtualenv --only-binary=:all: torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html 2>/dev/null || echo "PyG extras install failed - GCNConv may still work without them"

# Create directories
mkdir -p data/raw data/processed outputs/videos outputs/roi outputs/embeddings

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. .venv/bin/python data/generate_synthetic.py"
echo "  2. export GOOGLE_API_KEY='your-key'"
echo "  3. .venv/bin/python scripts/run_pipeline.py"
echo "  4. .venv/bin/python scripts/index_videos.py"
echo "  5. .venv/bin/python -m streamlit run src/dashboard/app.py"
