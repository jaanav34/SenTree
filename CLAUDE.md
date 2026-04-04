# SenTree — Resilience ROI Dashboard

Climate adaptation intelligence for sovereign wealth funds. 36-hour hackathon build.

## What This Is

A system that ingests ISIMIP3b climate data, detects tail-risk tipping points via momentum/volatility analysis, propagates systemic risk through a GNN, simulates interventions (mangroves, agriculture), renders downscaled video heatmaps, and makes everything searchable via native video embeddings.

**Win condition:** User types "Show where mangroves prevent collapse" → system returns video + ROI + highlighted region.

## Tech Stack

- **Data:** xarray, netCDF4, numpy, pandas
- **ML:** PyTorch, PyTorch Geometric (GCN)
- **Video:** matplotlib, ffmpeg-python, scipy (interpolation)
- **Search:** google-genai (Gemini embeddings), ChromaDB
- **UI:** Streamlit

## Repo Structure

```
src/data/          → ISIMIP loading, preprocessing, GDP/pop merge
src/tail_risk/     → Volatility, momentum, 95th-percentile flagging
src/graph/         → Grid→graph construction (PyG format)
src/model/         → 2-layer GCN, forward pass, inference
src/simulation/    → Intervention deltas, re-run through GNN, ROI calc
src/rendering/     → Downscaling (interpolation), heatmap frames, MP4 export
src/embedding/     → Video/frame embedding, ChromaDB index + query
src/dashboard/     → Streamlit app (search bar → video + ROI + risk flags)
```

## Key Commands

```bash
# Setup (mandatory venv)
./setup.sh                       # macOS/Linux
powershell -ExecutionPolicy Bypass -File .\\setup.ps1  # Windows

# Run (use venv python)
.venv/bin/python scripts/run_pipeline.py             # macOS/Linux
.venv/bin/python scripts/index_videos.py             # macOS/Linux
.venv/bin/python -m streamlit run src/dashboard/app.py  # macOS/Linux

.\\.venv\\Scripts\\python.exe scripts\\run_pipeline.py   # Windows
.\\.venv\\Scripts\\python.exe scripts\\index_videos.py   # Windows
.\\.venv\\Scripts\\python.exe -m streamlit run src\\dashboard\\app.py  # Windows

# System dependency
apt install ffmpeg  # or brew install ffmpeg
```

Note: Repo entrypoints refuse to run outside a virtualenv. To bypass (not recommended), set `SENTREE_ALLOW_NO_VENV=1`.

## Team Split

- **P1 (Scientist):** `src/data/` + `src/tail_risk/` → outputs `processed_data.pkl`
- **P2 (Architect):** `src/graph/` + `src/model/` + `src/simulation/` + `src/rendering/` → outputs MP4s + ROI CSVs
- **P3 (Product):** `src/embedding/` + `src/dashboard/` → wires search + UI

## Critical Decisions

- Region: Southeast Asia coastal (lat -10→25, lon 90→130)
- Scenario: SSP3-7.0 primary, SSP1-2.6 optional
- Time: 2015–2050 annual
- Downscaling: scipy interpolation + gaussian blur (NOT diffusion)
- Embeddings: Gemini 1.5 Pro multimodal → ChromaDB
- Fallback: frame-based CLIP embeddings if Gemini quota hit

## Don't

- Don't train the GNN for hours. Random init + 50 epochs max on tiny data.
- Don't use real diffusion models. Interpolation tricks only.
- Don't over-engineer. Copy-paste code, make it work, ship.
