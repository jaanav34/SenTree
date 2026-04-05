# SenTree

SenTree is a climate adaptation intelligence platform built for investor-style decision making.
It helps answer a concrete deployment question:

> If we allocate adaptation capital today, where should it go to avoid the most future loss — and how confident are we?

SenTree is not “just a model”. It’s an end-to-end decision workflow that turns climate time series into:

- **Tail-risk escalation detection** (where regimes are shifting, not just trending)
- **Systemic risk propagation** with a **Graph Neural Network (GNN)** (risk cascades across neighboring nodes)
- **Intervention simulation** with **Köppen–Geiger** climate-fit constraints (avoid biome-mismatch recommendations)
- **Resilience ROI + uncertainty bounds** (so “high ROI” isn’t confused with “high confidence”)
- **Evidence artifacts**: videos, maps, and searchable outputs for human review

The current build is optimized for a hackathon-style demo: clear decisions, explainable math, and fast visual evidence.

## Reproducibility / Runbook (Start Here If You’re New)

For a fully copy‑pasteable “from zero to working demo” guide (local + HPC/Slurm + port forwarding + video rendering),
read `instructions.md`.

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

At a high level, the pipeline (`scripts/run_pipeline.py`) is:

1. **Load climate data**
   - Real ISIMIP NetCDFs when present (daily → monthly/annual resampling)
   - Synthetic fallback for portable demos
2. **Detect tail-risk escalation**
   - EWMA smoothing + standardized momentum + rolling volatility
   - Self-exciting “clustered extremes” component (Hawkes intensity)
3. **Build a climate graph**
   - Each node is a grid cell; edges connect geographic neighbors
4. **Train a GNN to propagate systemic risk**
   - Predict node-level risk fields and how they evolve under interventions
5. **Simulate interventions**
   - Parameter deltas + Köppen–Geiger climate-fit allow/block rules
6. **Compute ROI + uncertainty**
   - Avoided loss proxy, ROI point estimate, confidence intervals
7. **Render evidence**
   - Core videos, per-intervention comparison videos, mega “grid” videos, and a static ROI target map PNG

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

## Research Foundations (What We Implemented)

SenTree’s math and visual evidence are grounded in four research building blocks. The Streamlit app contains a
plain-language walkthrough of these papers and their equations (see `src/dashboard/app.py`).

1) **Gurjar & Camp (2026) — EWMA tail-risk detection**
- SenTree uses EWMA smoothing to suppress short-lived noise, then measures standardized momentum and rolling volatility.
- Output: a per-node regime classification (**Baseline / Buildup / Surge**) and a core signal for tail-risk scoring.
- Implementation: `src/tail_risk/volatility.py`, `src/tail_risk/momentum.py`.

2) **Hawkes (1971) / Ogata (1988) — Self-exciting point processes**
- Extremes cluster (aftershocks, crisis cascades). SenTree models clustered climate extremes as a Hawkes-style intensity.
- Output: a normalized self-excitation signal that contributes to the composite tail-risk score.
- Implementation: `src/tail_risk/engine.py` (Hawkes intensity), combined into the composite score.

3) **Hess et al. (2023) — CycleGAN-inspired downscaling (adapted)**
- Climate + risk fields are often coarse. SenTree adapts Hess et al.’s key ideas into a lightweight deterministic
  downscaler: bicubic upsampling + gradient-aware texture + smoothing + physical constraints.
- Output: high-resolution frames for videos/maps without long GPU training runs.
- Implementation: `src/rendering/downscale.py` and video renderers in `src/rendering/render_video.py`.

4) **Ito et al. (2020) — FRA uncertainty + skill diagnostics**
- Climate impact models can be over-confident. SenTree uses FRA-inspired uncertainty inflation and decomposes uncertainty
  into precipitation/model/scenario components, producing honest ROI bounds.
- Output: ROI confidence intervals and an investor-facing confidence signal.
- Implementation: `src/simulation/roi.py`.

## Why This Is Impactful (What It Solves)

Most climate dashboards stop at “here’s a hazard map.” SenTree is built to support an actual allocation decision:

- **Early warning**: find locations entering unstable regimes (not just warm places).
- **Systemic view**: model risk propagation across space (cascades), not only per-cell scores.
- **Actionability**: simulate interventions and show their *counterfactual* effect (baseline vs intervention).
- **Honest uncertainty**: provide confidence bounds so high ROI doesn’t hide fragile assumptions.
- **Auditability**: generate videos/maps that allow humans to verify and communicate the story of each recommendation.

## Requirements

- Python 3.11+
- ffmpeg on PATH for MP4 generation
- Gemini API key only for semantic indexing/search (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)
- Node.js + npm only if you want the optional React GNN playback app (`apps/gnn-playback/`)

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
- `SENTREE_SAVE_RISK_SERIES_NPZ=1` save NPZ series for parallel video rendering later
- `SENTREE_KG_WORKERS=<n>` opt-in CPU parallelism for Köppen–Geiger (Linux/HPC)
- `SENTREE_FFMPEG_THREADS=<n>` ffmpeg encoding threads (cluster friendly)

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
