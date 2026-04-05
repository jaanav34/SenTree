# SenTree — Full Reproducibility Guide (Runbook)

This is a “from zero to demo” runbook intended for new contributors who have **no prior cluster context**.
It documents:

- How to set up Python + dependencies
- How to run the pipeline locally (synthetic data) and on HPC (ISIMIP NetCDFs)
- How to control runtime (coarsening, caching, CPU/GPU usage)
- How to render videos at scale (Slurm arrays + mega grid video)
- How to run the Streamlit dashboard + React GNN playback together and view them on your laptop via SSH tunnels
- Common failure modes and copy‑paste fixes

If you only want the quickstart, see `README.md`. This file is the “everything we wish we had on day one”.

---

## 0) High-level mental model

SenTree’s pipeline (`scripts/run_pipeline.py`) does:

1. Load climate data (ISIMIP if present, synthetic fallback for demos)
2. Resample daily → monthly/annual (for ISIMIP)
3. Compute tail-risk features and flags
4. Build a climate graph (nodes=grid cells, edges=neighbors)
5. Train a GNN to propagate systemic risk
6. Run interventions and compute ROI and uncertainty
7. Save artifacts (JSON/NPZ) + optionally render MP4s + a static ROI/target PNG

Artifacts are written under `outputs/` (gitignored by design).

---

## 1) Prerequisites

### 1.1 Python

- Python **3.11+** is recommended.
- Use the provided setup scripts:
  - macOS/Linux: `./setup.sh`
  - Windows PowerShell: `.\setup.ps1`

SenTree uses a virtualenv and enforces it via `sentree_venv.py` (scripts call `ensure_venv()`).

### 1.2 ffmpeg (required for MP4)

MP4 rendering uses Matplotlib + ffmpeg. If ffmpeg is missing, MP4 exports will fail.

- On macOS: `brew install ffmpeg`
- On Ubuntu: `sudo apt-get install ffmpeg`
- On Windows: install ffmpeg and make sure `ffmpeg` is on PATH, or set `SENTREE_FFMPEG_PATH`.

**Note (HPC):** On some clusters, the available `ffmpeg` build may not include GPU encoders (NVENC). SenTree currently targets CPU encoding and parallelizes via Slurm arrays.

### 1.3 Node.js + npm (optional; for React playback)

The React playback app lives in `apps/gnn-playback/`.

You need Node + npm:

- Local: install Node 18+ (22 works well).
- HPC: load a Node module, or add a bundled Node binary to your PATH (this repo may include a `node-v22...` folder in some environments).

The helper script `./scripts/run_gnn_playback.sh` will:
1) export playback JSON
2) `npm install` (first time)
3) `npm run dev`

---

## 2) Setup (local machine)

From repo root:

```bash
./setup.sh
```

Then (optional, but recommended for a first run) generate synthetic climate data:

```bash
.venv/bin/python data/generate_synthetic.py
```

Run the pipeline:

```bash
.venv/bin/python scripts/run_pipeline.py
```

Launch Streamlit:

```bash
.venv/bin/python -m streamlit run src/dashboard/app.py
```

Launch React playback (optional):

```bash
./scripts/run_gnn_playback.sh
```

---

## 3) Setup (HPC / Slurm)

### 3.1 Find your Slurm account

Some clusters require an explicit account with `sbatch -A <account>`.

Example (Gautschi):

```bash
sacctmgr -n -P show assoc where user=$USER format=Account,Partition,QOS
```

Then export it (used by wrapper scripts):

```bash
export SENTREE_SLURM_ACCOUNT=<your_account>
```

### 3.2 Run pipeline as a batch job

This repo includes Slurm job scripts under `jobs/`.

Typical submit:

```bash
sbatch -A "$SENTREE_SLURM_ACCOUNT" jobs/pipeline_ai_gpu.sbatch
```

Logs are written to `logs/` according to the `#SBATCH -o/-e` patterns inside the sbatch file.

Follow logs:

```bash
tail -f logs/pipeline_ai.<jobid>.out
tail -f logs/pipeline_ai.<jobid>.err
```

### 3.3 CPU usage vs “requested CPUs”

Requesting CPUs via Slurm (`#SBATCH -c N`) does not automatically make libraries use them.

For CPU-heavy steps, you often want these inside the job:

- `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`
- `MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK`
- `OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK`
- `NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK`

SenTree also supports opt-in CPU parallelism for Köppen–Geiger:

- `SENTREE_KG_WORKERS` (default `1`; set it to your CPU count if you want parallel KG on Linux/HPC)

Video encoding threads:

- `SENTREE_FFMPEG_THREADS` (used by the Matplotlib ffmpeg writer)

---

## 4) Data sources: ISIMIP vs synthetic

### 4.1 Synthetic (fast, portable)

Generate:

```bash
.venv/bin/python data/generate_synthetic.py
```

Run:

```bash
.venv/bin/python scripts/run_pipeline.py
```

### 4.2 ISIMIP (real NetCDF, heavier)

Put ISIMIP NetCDF files under `data/raw/` or point the pipeline at your raw directory.

Useful env var:

- `SENTREE_RAW_DIR=/path/to/isimip/raw`

When ISIMIP is found, the loader will print:

- number of NetCDF files found for `tas` and `pr`
- time range
- resampling steps (daily → monthly/annual)

Processed climate caches are written under `data/processed/` (gitignored).

---

## 5) Performance knobs (the ones you’ll actually use)

### 5.1 Region + resolution

Pipeline env vars:

- `SENTREE_REGION` (default: `se_asia`)
  - use `global` for global runs
- `SENTREE_COARSEN` (default: `1`)
  - `1` = full resolution (slow/large)
  - `2` or `4` = coarser (faster)
- `SENTREE_INTERVENTION_STRENGTH` (default: `1.0`)

### 5.2 Skip expensive steps during iteration

Render toggles:

- `SENTREE_RENDER_VIDEOS=0` disable all MP4 rendering
- `SENTREE_RENDER_CORE_VIDEOS=0` disable baseline/tail/KG videos
- `SENTREE_RENDER_COMPARISON_VIDEOS=0` disable per-intervention comparisons
- `SENTREE_RENDER_MAP_PNG=0` disable the static ROI PNG

Convenience toggles:

- `SENTREE_NO_VIDEOS=1` (equivalent to disabling all videos)
- `SENTREE_NO_COMPARISON_VIDEOS=1` (keep core videos, skip comparisons)

Time-series inference toggle:

- `SENTREE_COMPUTE_TIME_SERIES=0` to skip the baseline+intervention yearly inference series

### 5.3 Save risk series for later (enables parallel video rendering)

If you want to keep pipeline runtime low but still produce videos later:

- `SENTREE_SAVE_RISK_SERIES_NPZ=1`

This creates:

- `outputs/roi/risk_series/baseline.npz`
- `outputs/roi/risk_series/intervention_<key>.npz`

---

## 6) Video rendering at scale (recommended workflow for many interventions)

### 6.1 Render per-intervention comparison videos in parallel (Slurm array)

After you have risk series NPZs:

```bash
export SENTREE_SLURM_ACCOUNT=<account>
bash scripts/submit_render_comparisons.sh
```

This submits a Slurm job array where each task renders:

- `outputs/videos/comparison_<key>.mp4`

You can tail an individual task’s logs:

```bash
tail -f logs/render_cmp.<arrayjobid>_1.out
```

### 6.2 Mega “grid” video (all interventions at once)

Submit:

```bash
export SENTREE_SLURM_ACCOUNT=<account>
bash scripts/submit_render_megavideo.sh
```

Choose what the grid means:

- green risk reduction (baseline − intervention):
  - `SENTREE_MEGA_GRID_KIND=delta`
- absolute intervention risk (same scale as baseline):
  - `SENTREE_MEGA_GRID_KIND=intervention`

Example:

```bash
SENTREE_MEGA_GRID_KIND=intervention \
SENTREE_MEGA_OUT=outputs/videos/interventions_grid_intervention_risk.mp4 \
bash scripts/submit_render_megavideo.sh
```

---

## 7) Static ROI/target PNG (continents/borders + no embedded coordinate text)

The pipeline can render a static ROI/target map:

- `outputs/tail_risk_map.png`

Notes:

- The map no longer embeds “Target: (…E, …N)” annotation text.
- If Cartopy is installed, SenTree overlays land/coastlines/borders for context.
- Toggle borders overlay via:
  - `SENTREE_DRAW_BORDERS=0` to disable Cartopy overlay

---

## 8) Dashboard + React playback together (HPC → laptop)

### 8.1 Run both apps on the compute node with tmux

On the compute node (example hostname: `a242`):

```bash
tmux new -s sentree
```

Pane 1 (Streamlit):

```bash
cd ~/SenTree
streamlit run src/dashboard/app.py --server.port 8502 --server.address 127.0.0.1
```

Pane 2 (React/Vite):

```bash
cd ~/SenTree
./scripts/run_gnn_playback.sh
```

Detach:

```text
Ctrl-b d
```

### 8.2 Verify ports on the compute node

```bash
ss -ltnp | egrep ':8502|:4173|:4174'
```

Vite uses 4173 by default but will bump (4174, …) if 4173 is already taken.

### 8.3 SSH tunnels from your laptop

Forward Streamlit + Vite from your laptop to the compute node via the login node:

```bash
ssh -N -J <user>@gautschi.rcac.purdue.edu \
  -L 8502:127.0.0.1:8502 \
  -L 4173:127.0.0.1:4173 \
  <user>@a242.gautschi.rcac.purdue.edu
```

If Vite is on 4174, forward 4174 instead.

Open:

- Streamlit: `http://localhost:8502`
- React: `http://localhost:4173`

### 8.4 Embed React inside Streamlit

In Streamlit, click the `GNN Playback` tab and choose **Embedded React app**.

It defaults to:

- `http://localhost:4173/`

Override via:

- env `SENTREE_GNN_PLAYBACK_URL`
- or the text input in the UI

---

## 9) Troubleshooting (copy/paste)

### “npm was not found on PATH”

Load Node or add it to PATH. Example using a bundled Node folder:

```bash
export PATH="$PWD/node-v22.14.0-linux-x64/bin:$PATH"
node -v
npm -v
```

### “channel X: open failed: connect failed: Connection refused”

Your SSH tunnel is forwarding to a port where nothing is listening.

Fix:

- verify listeners on the compute node: `ss -ltnp | egrep ':8502|:4173'`
- forward to the *same node* where the services run
- forward the *correct port* (don’t forward 8502 → 8501 by accident)

### React playback stuck on “Loading playback data…”

The JSON can be huge if you export all nodes.

Use sampling env vars:

```bash
SENTREE_PLAYBACK_MAX_NODES=5000 SENTREE_PLAYBACK_MAX_EDGES=200000 ./scripts/run_gnn_playback.sh
```

### Streamlit port not available

Pick a different port:

```bash
streamlit run src/dashboard/app.py --server.port 8502 --server.address 127.0.0.1
```

---

## 10) What to commit vs what not to commit

By default:

- `outputs/` is gitignored (large generated files)
- `data/processed/` is gitignored (caches)

If you need to share results:

- share `outputs/roi/*.json` and any NPZs/videos via an artifact store, not git

