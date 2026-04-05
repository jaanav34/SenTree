# SenTree

SenTree is a climate adaptation intelligence platform for investor-style decision making.  
It combines:
- Graph neural network (GNN) climate risk propagation
- Tail-risk escalation detection
- Koppen-Geiger climate suitability filtering
- Intervention simulation and resilience ROI
- Native semantic search over simulation videos (Gemini Embedding 2, no RAG text middle layer)

The current build is optimized for a hackathon demo: clear decisions, explainable math, and fast visual evidence.

## Product Goal

SenTree answers one practical question:

**"If we allocate adaptation capital today, where do we put it to avoid the most future loss?"**

It is designed to feel like an investment decision cockpit:
- `Recommendation` for portfolio and intervention ranking
- `Evidence` for searchable videos, risk trajectories, and maps
- `Model` for GNN playback and mathematical foundations

## What Is New In The Current App

Recent work focused on product clarity and investor usability:
- Multi-level dashboard architecture (`Overview`, `Recommendation`, `Evidence`, `Model`)
- Recommendation workflows with shortlist + portfolio strategy
- Koppen-Geiger-aware intervention climate-fit explanations
- Investor capital allocation controls (currently `$5M` to `$100M`)
- AI resilience summary and decision-oriented KPI surface
- Embedded React GNN playback support with Streamlit fallback
- Semantic search wired to comparison videos for intervention-level retrieval
- Visual polish pass (hero layout, typography hierarchy, card system)

## Core Method

1. Load climate data (ISIMIP when available, synthetic fallback otherwise)
2. Compute tail-risk escalation features (momentum + volatility + self-excitation)
3. Build graph and train GNN for systemic risk propagation
4. Apply interventions with Koppen-Geiger compatibility constraints
5. Compute avoided-loss and resilience ROI with uncertainty terms
6. Render videos/maps and optionally index videos for semantic search

## Architecture Map

```text
Data (ISIMIP/Synthetic)
  -> Tail-Risk Engine (momentum, volatility, escalation)
  -> Graph Construction
  -> GNN Training + Inference
  -> Intervention Simulation (Koppen-aware rules)
  -> ROI + Uncertainty Metrics
  -> Video/Map Rendering
  -> Semantic Video Index (Gemini Embedding 2 + ChromaDB)
  -> Streamlit Dashboard
```

## Repository Layout

```text
data/
  generate_synthetic.py          Synthetic climate dataset generator
scripts/
  run_pipeline.py                End-to-end climate -> model -> ROI -> media pipeline
  index_videos.py                Semantic indexing into ChromaDB
  run_dashboard.sh/.ps1          Launch Streamlit dashboard
  run_gnn_playback.sh/.ps1       Launch React playback app
src/
  data/                          Loading, preprocessing, Koppen-Geiger logic
  tail_risk/                     Escalation metrics and scoring
  graph/                         Graph build
  model/                         GNN model and training
  simulation/                    Interventions + ROI logic
  rendering/                     Video rendering/downscaling
  embedding/                     Gemini/local embedders + vector DB
  dashboard/                     Streamlit app
apps/gnn-playback/               React playback frontend
```

## Requirements

- Python 3.11+
- ffmpeg on PATH for MP4 generation
- Gemini API key only for semantic indexing/search (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)

Install dependencies via setup scripts:

macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

Windows PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

## Quick Start

Run from repo root.

1. Generate synthetic data:
```bash
.venv/bin/python data/generate_synthetic.py
```

Windows:
```powershell
.\.venv\Scripts\python.exe data\generate_synthetic.py
```

2. Run pipeline:
```bash
.venv/bin/python scripts/run_pipeline.py
```

3. (Optional) index videos for semantic search:
```bash
.venv/bin/python scripts/index_videos.py
```

4. Launch dashboard:
```bash
.venv/bin/python -m streamlit run src/dashboard/app.py
```

Or launcher scripts:
```bash
./scripts/run_dashboard.sh
```

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_dashboard.ps1
```

5. Launch React GNN playback app (optional):
```bash
./scripts/run_gnn_playback.sh
```

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_gnn_playback.ps1
```

## Key Outputs

After `scripts/run_pipeline.py`:
- `outputs/roi/roi_results.json`
- `outputs/roi/risk_timeseries.json`
- `outputs/roi/opportunity_map.npz`
- `outputs/roi/gnn_training_history.npz`
- `outputs/videos/comparison_<intervention_key>.mp4`
- `outputs/videos/baseline_risk.mp4`
- `outputs/videos/tail_risk_escalation.mp4`
- `outputs/videos/climate_classification_shift.mp4`
- `outputs/tail_risk_map.png`

After `scripts/index_videos.py`:
- `outputs/embeddings/` (ChromaDB persistence)

## Dashboard Structure

The Streamlit dashboard is organized for narrative decision flow:

- `Overview`: mission snapshot + research foundations
- `Recommendation`:
  - `Brief`: top interventions, climate fit, portfolio strategy
  - `Comparison`: ROI/loss/risk-reduction chart and table
- `Evidence`:
  - `Search`: semantic retrieval over intervention videos
  - `Videos`: comparison/core/grid playback
  - `Risk Over Time`: intervention trajectory analysis
  - `Map`: opportunity and tail-risk geography
- `Model`:
  - `GNN Playback`: embedded React app or Streamlit fallback
  - `Math Foundations`: equations and method explanations

## Semantic Search Notes

- Uses Gemini Embedding 2 via `src/embedding/gemini_embedder.py`
- Performs native vector similarity over video representations
- Avoids transcript-RAG dependency for retrieval behavior
- Works best after comparison videos are rendered and indexed

## Configuration Notes

Useful env flags:
- `SENTREE_RENDER_VIDEOS=0` disable all video rendering
- `SENTREE_RENDER_COMPARISON_VIDEOS=0` disable per-intervention videos
- `SENTREE_RENDER_CORE_VIDEOS=0` disable baseline/tail/KG core videos
- `SENTREE_RENDER_MAP_PNG=0` disable `tail_risk_map.png`

## Current Scope And Caveats

- Prototype tuned for fast demo iteration, not production hardening
- ROI and confidence are simulation outputs and should be treated as decision support, not financial guarantees
- Search quality depends on rendered video quality and index coverage
- Running in `.venv` is enforced by `sentree_venv.py`

## One-Command Flow (Typical)

```bash
./setup.sh
.venv/bin/python data/generate_synthetic.py
.venv/bin/python scripts/run_pipeline.py
.venv/bin/python scripts/index_videos.py
.venv/bin/python -m streamlit run src/dashboard/app.py
```
