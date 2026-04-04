#!/bin/bash
set -e

echo "=== SenTree Setup ==="

python -m venv .venv
echo "Activating virtual environment..."
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# PyG CPU install (safe for hackathon)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html 2>/dev/null || echo "PyG extras install failed - GCNConv may still work without them"

# Create directories
mkdir -p data/raw data/processed outputs/videos outputs/roi outputs/embeddings

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. python data/generate_synthetic.py"
echo "  2. export GOOGLE_API_KEY='your-key'"
echo "  3. python scripts/run_pipeline.py"
echo "  4. python scripts/index_videos.py"
echo "  5. streamlit run src/dashboard/app.py"
