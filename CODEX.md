# SenTree — Codex Handoff / Runbook (April 4, 2026)

This file is written to let a *new* Codex / ChatGPT conversation pick up the SenTree project instantly, with no missing context.

It covers:
- What SenTree is and how the pipeline works end-to-end
- The exact ISIMIP dataset we downloaded (tas + pr) and where it lives on Gautschi
- The Slurm jobs we used (CPU + GPU), and the pitfalls we hit
- How we fixed CUDA/PyTorch, ffmpeg, and Streamlit video playback
- How to run “SE Asia demo” vs “Global” (and why coarsening matters)
- How to confirm you are using **real** ISIMIP data (not synthetic)
- Common failure modes and copy‑paste fixes

---

## 0) Glossary

- **Gautschi**: Purdue RCAC cluster (Slurm). Login nodes like `login01`. Compute nodes like `a243` (CPU) or `h003` (AI/H100).
- **Slurm**: job scheduler. `sbatch` submits non-interactive jobs; `srun --pty bash -l` gives interactive shells on compute nodes.
- **ISIMIP3b**: Inter‑Sectoral Impact Model Intercomparison Project climate input data.
- **tas**: near-surface air temperature.
- **pr**: precipitation (flux).
- **SE Asia subset**: the original demo region (lat ~[-10, 25], lon ~[90, 130]).
- **Global run**: the whole world grid (lat 360 x lon 720 at 0.5°). Too big at full res, so we use **coarsening**.

---

## 1) Repo layout (high level)

- `scripts/run_pipeline.py`: end-to-end pipeline (load data → tail risk → graph → GNN → interventions → ROI → videos → maps).
- `src/data/load_isimip.py`: loads ISIMIP NetCDFs, resamples daily→annual, converts units, and caches a pickle.
- `src/model/gnn.py`: GNN (GAT/GCN hybrid) and training/inference helpers. Updated to move model/data to CUDA automatically.
- `src/simulation/interventions.py`: intervention definitions (mangroves, regenerative ag).
- `src/simulation/run_simulations.py`: applies interventions to features and reruns model inference.
- `src/rendering/render_video.py`: video rendering via Matplotlib animations (`.mp4` requires ffmpeg).
- `src/dashboard/app.py`: Streamlit dashboard. Updated so videos render via bytes (important on clusters).
- `outputs/`: generated artifacts:
  - `outputs/videos/*.mp4`
  - `outputs/roi/roi_results.json`
  - `outputs/roi/risk_timeseries.json`
  - `outputs/roi/opportunity_map.npz`
  - `outputs/tail_risk_map.png`

---

## 2) What dataset we downloaded (real ISIMIP)

We downloaded **ISIMIP3b bias-adjusted climate input data**:
- Scenario: `ssp370`
- Model: `GFDL-ESM4`
- Temporal resolution: **daily**
- Spatial resolution: **global 0.5°**
- Variables:
  - `tas` (temperature, units in NetCDF: Kelvin)
  - `pr` (precipitation, units in NetCDF: `kg m-2 s-1`)
- Files are split by decade:
  - `*_2015_2020.nc`, `*_2021_2030.nc`, …, `*_2091_2100.nc`

### Unit conversions in code

In `src/data/load_isimip.py`:
- `tas` is converted from **K → °C** by subtracting 273.15.
- `pr` is converted from **kg m-2 s-1 → mm/day** by multiplying by 86400 (because 1 kg/m² = 1 mm water equivalent).

---

## 3) Where the raw ISIMIP files live (Gautschi)

On Gautschi, we ended up with the raw NetCDF files in a “sibling” scratch folder, not inside the repo:

- Repo root:
  - `/scratch/gautschi/shah958/sentree/SenTree`
- Raw data folder:
  - `/scratch/gautschi/shah958/sentree/data/raw`

To make the repo see the raw files, we created a symlink:

```bash
cd /scratch/gautschi/shah958/sentree/SenTree
mkdir -p data
rm -f data/raw
ln -s ../data/raw data/raw
```

Then `data/raw/*.nc` resolves to `../data/raw/*.nc`.

---

## 4) How `load_isimip.py` decides “real vs synthetic”

### Real path

`src/data/load_isimip.py` tries to find `tas` NetCDFs (required) and `pr` NetCDFs (optional) under a list of candidate directories:

- `SENTREE_RAW_DIR` (if set)
- `raw_dir` argument (default `data/raw`)
- repo-relative `data/raw` and repo-parent `data/raw`
- `cwd/data/raw` and `cwd/../data/raw`

When it finds NetCDFs, it prints something like:

```text
Using raw_dir=...
Found 9 ISIMIP tas NetCDF files...
Found 9 ISIMIP pr NetCDF files...
```

### Synthetic fallback

If no NetCDFs are found:
- For `region="se_asia"` it will fall back to synthetic.
- For `region="global"` it **raises an error** (to prevent silently “faking” a global run).

This change was added because we accidentally ran a “global” pipeline that silently used synthetic data and misled us.

---

## 5) Processed cache files (important)

The loader caches processed annual data as a pickle in `data/processed/`.

It is **region + coarsen** specific:

- `data/processed/climate_data_se_asia_c1.pkl`
- `data/processed/climate_data_global_c4.pkl`
- etc.

This prevents overwriting the SE Asia cache when you run global, and prevents a synthetic run from overwriting real data.

If you suspect the wrong cache is being used, delete it and rerun:

```bash
rm -f data/processed/climate_data_*.pkl
```

---

## 6) Running the pipeline (local vs Gautschi)

### Local Windows

Typically:
```powershell
python scripts/run_pipeline.py
streamlit run src/dashboard/app.py
```

### Gautschi (Slurm)

We use sbatch scripts under `jobs/` on Gautschi (these were created manually on the cluster):
- `jobs/pipeline_ai_gpu.sbatch` (AI partition, H100)
- There may also be CPU versions like `jobs/pipeline_cpu.sbatch`

The pipeline writes outputs into the repo `outputs/` directory.

---

## 7) Global runs: why coarsening is required

Raw global ISIMIP resolution is **360×720 = 259,200 nodes**.

The SenTree graph step is kNN-like (`k=8`) and the GNN training scales with nodes+edges. At full resolution it becomes extremely heavy (memory + time).

So we run global at a coarser grid via `coarsen=N`:
- `coarsen=4` → ~`90×180 = 16,200 nodes` (practical for hackathon runs)
- `coarsen=2` → ~`180×360 = 64,800 nodes` (much heavier)

---

## 8) Environment variables (Gautschi)

These are read in `scripts/run_pipeline.py`:

- `SENTREE_REGION`
  - `se_asia` (default)
  - `global`
- `SENTREE_COARSEN`
  - integer factor (default `1`)
  - recommended for global: `4`
- `SENTREE_INTERVENTION_STRENGTH`
  - scales intervention deltas (default `1.0`)
  - recommended for “show impact”: `2.0` or `3.0`
- `SENTREE_RAW_DIR`
  - optional absolute path to the raw NetCDF directory if symlinks aren’t used
- `SENTREE_FFMPEG_PATH` / `SENTREE_FFMPEG`
  - path to an ffmpeg executable (used by Matplotlib writer)

Example global run:

```bash
export SENTREE_REGION=global
export SENTREE_COARSEN=4
export SENTREE_INTERVENTION_STRENGTH=2.0
sbatch jobs/pipeline_ai_gpu.sbatch
```

---

## 9) Interventions “not doing anything”: what changed

Originally interventions sometimes looked nearly identical to baseline.

We added:
- A scalable `strength` multiplier to `apply_intervention(...)` in `src/simulation/run_simulations.py`.
- A coastal-only mask that can use `coastal_factor` (works globally) rather than relying only on longitude thresholds.

This makes the intervention effect more visible and more geographically sensible on global maps.

---

## 10) GPU: what we fixed (CUDA / PyTorch / torch_geometric)

### The symptom

On GPU nodes:
- `nvidia-smi` worked
- but `torch.cuda.is_available()` was `False`

The root cause was a mismatched PyTorch build (`cu130`) relative to the cluster driver/runtime (CUDA 12.x).

### The fix (performed on an interactive AI GPU node)

1) Get an interactive GPU shell:
```bash
srun -A mlp -p ai --gres=gpu:1 -c 14 --mem=32G -t 00:20:00 --pty bash -l
```

2) Load modules + venv:
```bash
module --force purge
module load modtree/gpu
module load gcc/14.1.0
module load cuda/12.6.1
source .venv/bin/activate
```

3) Replace torch with a CUDA 12.8 wheel:
```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

4) Verify:
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

5) Verify torch_geometric still works:
```bash
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

### Code change to actually use GPU

We updated `src/model/gnn.py` so:
- training calls `model.to(cuda)` and `data.to(cuda)` automatically when CUDA is available
- prediction moves `data` to the model’s device and returns CPU numpy arrays

Without this, even a correct GPU torch install would still run on CPU.

---

## 11) ffmpeg: how we enabled MP4 rendering on Gautschi

### Problem

`src/rendering/render_video.py` saves `.mp4` using ffmpeg. On Gautschi there is no `ffmpeg` module, so the pipeline failed at video rendering.

### Installed a static ffmpeg in scratch

```bash
mkdir -p $SCRATCH/tools
cd $SCRATCH/tools
wget -O ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg.tar.xz
cd ffmpeg-*-amd64-static
./ffmpeg -version
```

### Permanent PATH + env var

Add to `~/.bashrc`:

```bash
export PATH="$SCRATCH/tools/ffmpeg-7.0.2-amd64-static:$PATH"
export SENTREE_FFMPEG_PATH="$SCRATCH/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"
```

Then:
```bash
source ~/.bashrc
which ffmpeg
ffmpeg -version
```

### Slurm note

Batch jobs don’t always source your interactive `~/.bashrc`.
If MP4 rendering fails in a batch job, add `source ~/.bashrc` to the `.sbatch` file or export the vars inside the script.

---

## 12) Streamlit on Gautschi (and why videos didn’t load)

### Run Streamlit on a compute node

```bash
srun -A mlp -p cpu -c 4 --mem=16G -t 02:00:00 --pty bash -l
module --force purge
module load modtree/cpu
module load gcc/14.1.0
module load python/3.11.9
source .venv/bin/activate
streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

### SSH port forwarding (on your laptop)

If Streamlit runs on node `a243`:

```powershell
ssh -L 8501:a243.gautschi.rcac.purdue.edu:8501 shah958@gautschi.rcac.purdue.edu
```

Then open:
- `http://localhost:8501`

### Why videos didn’t load (fixed)

`st.video("outputs/videos/foo.mp4")` treats strings like URLs and may not serve cluster filesystem files properly over a tunnel.

We updated `src/dashboard/app.py` to read the mp4 bytes and pass them into `st.video(...)`:
- `st.video(Path(path).read_bytes(), format="video/mp4")`

This makes videos render reliably from the cluster filesystem.

---

## 13) How to confirm you are using real ISIMIP (fast checks)

### A) Check for the “synthetic” warning in pipeline logs

Bad (synthetic):
```text
WARNING: No ISIMIP data found. Generating synthetic data.
```

Good (real):
```text
Using raw_dir=...
Found 9 ISIMIP tas NetCDF files...
Found 9 ISIMIP pr NetCDF files...
```

### B) Confirm the raw NetCDFs are visible

```bash
ls -1 data/raw/*ssp370_tas_global_daily_*.nc | wc -l
ls -1 data/raw/*ssp370_pr_global_daily_*.nc | wc -l
```

Expect `9` and `9`.

### C) Inspect the processed pickle values

```bash
python - <<'PY'
import pickle, numpy as np, glob
p = sorted(glob.glob("data/processed/climate_data_*.pkl"))
print("pickles:", p[-5:])
d = pickle.load(open(p[-1], "rb"))
print("years:", int(d["years"][0]), "to", int(d["years"][-1]), "n=", len(d["years"]))
print("tas min/max:", float(np.min(d["tas"])), float(np.max(d["tas"])))
print("pr  min/max:", float(np.min(d["pr"])), float(np.max(d["pr"])))
print("grid:", len(d["lats"]), "x", len(d["lons"]))
PY
```

---

## 14) Known “gotchas” and fixes

### “I set env vars, but the job ignored them”

Slurm `sbatch` jobs inherit environment variables from the submission shell *by default*, but some clusters or scripts override this.

If needed, submit with:
```bash
sbatch --export=ALL jobs/pipeline_ai_gpu.sbatch
```

Or hardcode exports at the top of the sbatch script.

### “It still says synthetic even though files exist”

Almost always one of:
- you’re using a cached pickle from a previous synthetic run → delete `data/processed/climate_data_*.pkl`
- the job started before you `git pull`-ed the fix
- `data/raw` doesn’t point to the raw directory (fix symlink)
- you ran from a different working directory than you think

### “`rg` not found on Gautschi”

Use `grep`:
```bash
grep -R "st.video" -n src/dashboard/app.py
```

### “ffmpeg works on login node but not in sbatch”

Add:
```bash
source ~/.bashrc
```
inside the sbatch script.

---

## 15) What to do next (suggested)

- Rerun a **real ISIMIP global** pipeline (coarsen=4) and confirm logs show NetCDF loading.
- Update the Streamlit dashboard to display which cache file is loaded (e.g., show `region/coarsen/raw_dir` in the UI).
- Consider a “global‑aware” intervention mask:
  - mangroves should key off `coastal_factor` (already added) and optionally latitude bands / known coastlines
  - regen ag should use a land mask (currently we do not have a land/sea mask; we treat everything as land)

---

## 16) Security / secrets

- Do not commit `.env`.
- API keys (Gemini) live in `.env` on local machines; don’t paste keys into logs or commits.

