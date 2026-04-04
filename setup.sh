#!/bin/bash
set -e

echo "=== SenTree Setup ==="

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY" 1>&2
  exit 1
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install --require-virtualenv -r requirements.txt

# PyG CPU install (safe for hackathon)
"$VENV_PY" -m pip install --require-virtualenv torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html 2>/dev/null || echo "PyG extras install failed - GCNConv may still work without them"

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
