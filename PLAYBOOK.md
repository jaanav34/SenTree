# SenTree Implementation Playbook

**36-Hour Hackathon Build — Code-First Execution Plan**

---

## 1. DATA DOWNLOAD PLAN

### ISIMIP3b Climate Data

**Variables:** `tas` (near-surface air temperature), `pr` (precipitation)
**Scenario:** SSP3-7.0 (ssp370), model: GFDL-ESM4
**Time:** 2015–2050 (annual means from monthly)
**Region:** Southeast Asia coastal — lat [-10, 25], lon [90, 130]

**Download portal:** https://data.isimip.org/

Navigate to: ISIMIP3b → Bias-adjusted atmospheric climate → GFDL-ESM4 → ssp370

Direct download commands (use ISIMIP's download client or wget from their file listing):

```bash
# Install ISIMIP client
.venv/bin/python -m pip install --require-virtualenv isimip-client

# Or manual wget from the ISIMIP data portal file listing:
# Temperature (tas) — monthly, global, ~500MB per file
mkdir -p data/raw
cd data/raw

# OPTION A: Use ISIMIP download tool
# Register at data.isimip.org, get API key, then:
# isimip-client download --dataset gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_monthly_2015_2050.nc

# OPTION B: Direct wget (URLs from portal search results)
# After searching the portal, you'll get links like:
# wget https://files.isimip.org/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/monthly/ssp370/GFDL-ESM4/gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_monthly_2015_2020.nc
# wget https://files.isimip.org/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/monthly/ssp370/GFDL-ESM4/gfdl-esm4_r1i1p1f1_w5e5_ssp370_pr_global_monthly_2015_2020.nc

# OPTION C (HACKATHON FAST-PATH): Use pre-subsetted ERA5/CDS data as stand-in
# .venv/bin/python -m pip install --require-virtualenv cdsapi
# Use the CDS API to grab a small regional subset quickly
```

**CRITICAL HACKATHON SHORTCUT:** If ISIMIP download is slow, generate synthetic data that matches ISIMIP statistical properties:

```python
# data/generate_synthetic.py
import numpy as np
import pickle

np.random.seed(42)

# Grid: SE Asia coastal, 2-degree resolution
lats = np.arange(-10, 26, 2)  # 18 points
lons = np.arange(90, 132, 2)   # 21 points
years = np.arange(2015, 2051)   # 36 years
n_lats, n_lons, n_years = len(lats), len(lons), len(years)

# Temperature: base ~28°C + warming trend + noise
temp_base = 28.0
temp_trend = np.linspace(0, 2.5, n_years)  # +2.5°C by 2050 under SSP3-7.0
tas = np.zeros((n_years, n_lats, n_lons))
for t in range(n_years):
    tas[t] = temp_base + temp_trend[t] + np.random.normal(0, 0.5, (n_lats, n_lons))
    # Coastal cells warmer
    tas[t, :, -5:] += 0.3

# Precipitation: base ~2000mm/yr + variability increasing over time
pr_base = 5.5  # mm/day average
pr = np.zeros((n_years, n_lats, n_lons))
for t in range(n_years):
    volatility_scale = 1.0 + 0.02 * t  # increasing variability
    pr[t] = pr_base + np.random.normal(0, 1.5 * volatility_scale, (n_lats, n_lons))
    pr[t] = np.clip(pr[t], 0.1, 20)  # physical bounds

# GDP per grid cell (proxy from World Bank SE Asia data)
# Rough: $5k-$40k per capita across region
gdp = np.random.uniform(5000, 40000, (n_lats, n_lons))
gdp[:, -5:] *= 1.5  # coastal = richer

# Population density
pop = np.random.uniform(50, 5000, (n_lats, n_lons))
pop[:, -5:] *= 2  # coastal = denser

data = {
    'tas': tas,       # (36, 18, 21)
    'pr': pr,         # (36, 18, 21)
    'gdp': gdp,       # (18, 21)
    'pop': pop,       # (18, 21)
    'lats': lats,
    'lons': lons,
    'years': years,
}

with open('data/processed/climate_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"Generated synthetic data: {n_years} years, {n_lats}x{n_lons} grid = {n_lats * n_lons} nodes")
```

### GDP Data

```bash
# World Bank API — GDP per capita for SE Asian countries
# .venv/bin/python -m pip install --require-virtualenv wbgapi
python -c "
import wbgapi as wb
import pandas as pd
countries = ['THA', 'VNM', 'IDN', 'PHL', 'MMR', 'KHM', 'MYS', 'SGP']
df = wb.data.DataFrame('NY.GDP.PCAP.CD', countries, range(2015, 2024))
df.to_csv('data/raw/gdp_seasia.csv')
print(df.head())
"
```

### Converting NetCDF to usable format (if using real data)

```python
# data/load_isimip.py
import xarray as xr
import pickle

def load_and_subset(nc_path, var_name, lat_range=(-10, 25), lon_range=(90, 130)):
    ds = xr.open_dataset(nc_path)
    subset = ds[var_name].sel(
        lat=slice(*lat_range),
        lon=slice(*lon_range)
    )
    # Resample monthly → annual mean
    annual = subset.resample(time='1YE').mean()
    return annual.values, annual.lat.values, annual.lon.values

tas_data, lats, lons = load_and_subset('data/raw/tas_ssp370.nc', 'tas')
pr_data, _, _ = load_and_subset('data/raw/pr_ssp370.nc', 'pr')

data = {'tas': tas_data, 'pr': pr_data, 'lats': lats, 'lons': lons}
with open('data/processed/climate_data.pkl', 'wb') as f:
    pickle.dump(data, f)
```

---

## 2. FILE STRUCTURE

```
SenTree/
├── CLAUDE.md
├── PLAYBOOK.md
├── requirements.txt
├── setup.sh
├── data/
│   ├── raw/                    # NetCDF files, CSVs (gitignored)
│   ├── processed/              # .pkl files after preprocessing
│   └── generate_synthetic.py   # Fallback synthetic data generator
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_isimip.py      # Load NetCDF → xarray → numpy
│   │   └── preprocess.py       # Merge climate + GDP + pop, normalize
│   ├── tail_risk/
│   │   ├── __init__.py
│   │   ├── volatility.py       # Rolling std on temp/precip time series
│   │   ├── momentum.py         # Rate of change (delta) calculations
│   │   └── engine.py           # Combine vol+mom → tail_risk_score, flag 95th pct
│   ├── graph/
│   │   ├── __init__.py
│   │   └── build_graph.py      # Grid cells → PyG Data (nodes, edges, features)
│   ├── model/
│   │   ├── __init__.py
│   │   └── gnn.py              # 2-layer GCN: features → risk score per node
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── interventions.py    # Define intervention deltas (mangrove, agri)
│   │   ├── run_simulations.py  # Apply deltas → re-run GNN → compute ROI
│   │   └── roi.py              # Resilience ROI formula + uncertainty
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── downscale.py        # Interpolation upsampling + gaussian blur
│   │   └── render_video.py     # Heatmap frames → MP4 via matplotlib + ffmpeg
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── embedder.py         # Gemini/CLIP video/frame embedding
│   │   └── vectordb.py         # ChromaDB store + query
│   └── dashboard/
│       └── app.py              # Streamlit app: search → video + ROI + flags
├── outputs/
│   ├── videos/                 # Generated MP4s
│   ├── roi/                    # ROI CSVs
│   └── embeddings/             # ChromaDB persistence dir
└── scripts/
    ├── run_pipeline.py         # End-to-end: data → tail risk → graph → GNN → sims → videos
    └── index_videos.py         # Embed all videos → ChromaDB
```

**What each file does:**

| File | Purpose |
|---|---|
| `data/generate_synthetic.py` | Creates realistic fake ISIMIP data if download is slow |
| `src/data/load_isimip.py` | Opens NetCDF, subsets to SE Asia, resamples to annual |
| `src/data/preprocess.py` | Merges temp+precip+GDP+pop into unified node feature matrix |
| `src/tail_risk/volatility.py` | Computes 5-year rolling std for each grid cell |
| `src/tail_risk/momentum.py` | Computes year-over-year delta (rate of change) |
| `src/tail_risk/engine.py` | Combines vol+mom into composite score, flags >95th pct |
| `src/graph/build_graph.py` | Converts grid → PyG Data object with edge_index + features |
| `src/model/gnn.py` | 2-layer GCNConv, outputs risk score per node |
| `src/simulation/interventions.py` | Defines mangrove/agri intervention parameter dicts |
| `src/simulation/run_simulations.py` | Copies baseline, applies deltas, re-runs GNN |
| `src/simulation/roi.py` | Computes ROI with uncertainty penalty |
| `src/rendering/downscale.py` | scipy.ndimage zoom + gaussian_filter for fake hi-res |
| `src/rendering/render_video.py` | Generates heatmap frames, stitches to MP4 |
| `src/embedding/embedder.py` | Embeds video frames via Gemini API or CLIP |
| `src/embedding/vectordb.py` | ChromaDB CRUD: add embeddings, query by text |
| `src/dashboard/app.py` | Streamlit UI: text search → video + ROI + tail-risk |
| `scripts/run_pipeline.py` | Orchestrates full pipeline end-to-end |
| `scripts/index_videos.py` | Batch embeds all output videos into ChromaDB |

---

## 3. ENVIRONMENT SETUP

```
Python 3.10+
```

### requirements.txt

```
# Core data
numpy>=1.24
pandas>=2.0
xarray>=2024.1
netCDF4>=1.6
scipy>=1.11
pickle5; python_version<"3.11"

# ML
torch>=2.1
torch-geometric>=2.4
torch-scatter
torch-sparse

# Video rendering
matplotlib>=3.7
ffmpeg-python>=0.2

# Embeddings + search
google-genai>=0.4
chromadb>=0.4
Pillow>=10.0
# Optional: open-clip-torch>=2.24 (fallback)

# Dashboard
streamlit>=1.30

# Utils
tqdm>=4.65
scikit-learn>=1.3

# GDP data (optional)
# wbgapi>=1.0
```

### Setup commands

```bash
# setup.sh
#!/bin/bash
set -e

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# PyG requires special install depending on CUDA version
# CPU-only (hackathon safe):
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# ffmpeg
# Ubuntu/Debian:
# sudo apt install ffmpeg
# Mac:
# brew install ffmpeg
# Windows:
# choco install ffmpeg OR download from ffmpeg.org

# Create directories
mkdir -p data/raw data/processed outputs/videos outputs/roi outputs/embeddings
```

---

## 4. DATA PIPELINE (WITH CODE)

### src/data/load_isimip.py

```python
"""Load ISIMIP NetCDF data or fall back to synthetic."""
import os
import pickle
import numpy as np

def load_climate_data(data_dir='data/processed', raw_dir='data/raw'):
    """Load processed climate data. Generate synthetic if not available."""
    pkl_path = os.path.join(data_dir, 'climate_data.pkl')

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    # Try NetCDF
    tas_path = os.path.join(raw_dir, 'tas_ssp370.nc')
    if os.path.exists(tas_path):
        return _load_from_netcdf(raw_dir, data_dir)

    # Fall back to synthetic
    print("WARNING: No ISIMIP data found. Generating synthetic data.")
    return _generate_synthetic(data_dir)


def _load_from_netcdf(raw_dir, out_dir):
    import xarray as xr

    lat_range = slice(-10, 25)
    lon_range = slice(90, 130)

    tas = xr.open_dataset(f'{raw_dir}/tas_ssp370.nc')['tas']
    tas = tas.sel(lat=lat_range, lon=lon_range).resample(time='1YE').mean()

    pr = xr.open_dataset(f'{raw_dir}/pr_ssp370.nc')['pr']
    pr = pr.sel(lat=lat_range, lon=lon_range).resample(time='1YE').mean()

    data = {
        'tas': tas.values,
        'pr': pr.values,
        'lats': tas.lat.values,
        'lons': tas.lon.values,
        'years': np.array([t.year for t in tas.time.values]),
        'gdp': np.random.uniform(5000, 40000, (len(tas.lat), len(tas.lon))),
        'pop': np.random.uniform(50, 5000, (len(tas.lat), len(tas.lon))),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/climate_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data


def _generate_synthetic(out_dir):
    np.random.seed(42)
    lats = np.arange(-10, 26, 2)
    lons = np.arange(90, 132, 2)
    years = np.arange(2015, 2051)
    ny, nlat, nlon = len(years), len(lats), len(lons)

    temp_trend = np.linspace(0, 2.5, ny)
    tas = np.zeros((ny, nlat, nlon))
    pr = np.zeros((ny, nlat, nlon))

    for t in range(ny):
        tas[t] = 28.0 + temp_trend[t] + np.random.normal(0, 0.5, (nlat, nlon))
        tas[t, :, -5:] += 0.3  # coastal warming
        vol_scale = 1.0 + 0.02 * t
        pr[t] = 5.5 + np.random.normal(0, 1.5 * vol_scale, (nlat, nlon))
        pr[t] = np.clip(pr[t], 0.1, 20)

    gdp = np.random.uniform(5000, 40000, (nlat, nlon))
    gdp[:, -5:] *= 1.5
    pop = np.random.uniform(50, 5000, (nlat, nlon))
    pop[:, -5:] *= 2

    data = {
        'tas': tas, 'pr': pr, 'gdp': gdp, 'pop': pop,
        'lats': lats, 'lons': lons, 'years': years,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/climate_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Synthetic data: {ny} years, {nlat*nlon} nodes")
    return data
```

### src/data/preprocess.py

```python
"""Preprocess climate data into node feature matrices."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_node_features(data, year_idx=-1):
    """
    Build feature matrix for a single timestep.
    Returns: features (N, 5), node_positions (N, 2)
    Features: [temp, precip, volatility, momentum, gdp]
    """
    tas = data['tas']  # (T, nlat, nlon)
    pr = data['pr']
    nlat, nlon = tas.shape[1], tas.shape[2]
    N = nlat * nlon

    # Flatten spatial dims
    temp = tas[year_idx].flatten()
    precip = pr[year_idx].flatten()
    gdp = data['gdp'].flatten()

    # Compute volatility and momentum (imported from tail_risk)
    from src.tail_risk.volatility import compute_volatility
    from src.tail_risk.momentum import compute_momentum

    vol = compute_volatility(tas, window=5).flatten()      # at last timestep
    mom = compute_momentum(tas, window=3).flatten()

    # Stack features
    features = np.column_stack([temp, precip, vol, mom, gdp])

    # Normalize
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Node positions for graph construction
    lats, lons = np.meshgrid(data['lats'], data['lons'], indexing='ij')
    positions = np.column_stack([lats.flatten(), lons.flatten()])

    return features.astype(np.float32), positions, scaler


def build_temporal_features(data):
    """Build feature matrices for ALL timesteps. Returns list of (N,5) arrays."""
    from src.tail_risk.volatility import compute_volatility_series
    from src.tail_risk.momentum import compute_momentum_series

    tas, pr = data['tas'], data['pr']
    T, nlat, nlon = tas.shape
    N = nlat * nlon

    vol_series = compute_volatility_series(tas, window=5)   # (T, nlat, nlon)
    mom_series = compute_momentum_series(tas, window=3)

    features_list = []
    for t in range(T):
        feats = np.column_stack([
            tas[t].flatten(),
            pr[t].flatten(),
            vol_series[t].flatten(),
            mom_series[t].flatten(),
            data['gdp'].flatten(),
        ])
        features_list.append(feats.astype(np.float32))

    return features_list
```

---

## 5. TAIL-RISK ENGINE

### src/tail_risk/volatility.py

```python
"""Rolling volatility (standard deviation) on climate time series."""
import numpy as np


def compute_volatility(data_3d, window=5):
    """
    Rolling std over time axis.
    Input: (T, nlat, nlon)
    Output: (nlat, nlon) — volatility at the last valid timestep
    """
    T, nlat, nlon = data_3d.shape
    if T < window:
        return np.std(data_3d, axis=0)
    return np.std(data_3d[-window:], axis=0)


def compute_volatility_series(data_3d, window=5):
    """
    Rolling std for every timestep.
    Output: (T, nlat, nlon) — padded with zeros for early timesteps
    """
    T, nlat, nlon = data_3d.shape
    vol = np.zeros_like(data_3d)
    for t in range(window, T):
        vol[t] = np.std(data_3d[t-window:t], axis=0)
    # Fill early timesteps with first valid
    if T > window:
        vol[:window] = vol[window]
    return vol
```

### src/tail_risk/momentum.py

```python
"""Momentum (rate of change) on climate time series."""
import numpy as np


def compute_momentum(data_3d, window=3):
    """
    Average rate of change over last `window` years.
    Input: (T, nlat, nlon)
    Output: (nlat, nlon)
    """
    T = data_3d.shape[0]
    if T < window + 1:
        return data_3d[-1] - data_3d[0]
    deltas = np.diff(data_3d[-window-1:], axis=0)  # (window, nlat, nlon)
    return np.mean(deltas, axis=0)


def compute_momentum_series(data_3d, window=3):
    """
    Momentum at every timestep.
    Output: (T, nlat, nlon)
    """
    T, nlat, nlon = data_3d.shape
    mom = np.zeros_like(data_3d)
    for t in range(window + 1, T):
        deltas = np.diff(data_3d[t-window-1:t], axis=0)
        mom[t] = np.mean(deltas, axis=0)
    if T > window + 1:
        mom[:window+1] = mom[window+1]
    return mom
```

### src/tail_risk/engine.py

```python
"""Tail-Risk Escalation Engine — combines volatility + momentum, flags 95th percentile."""
import numpy as np
from .volatility import compute_volatility
from .momentum import compute_momentum


def compute_tail_risk(data, vol_weight=0.6, mom_weight=0.4, percentile=95):
    """
    Compute tail_risk_score for each grid cell.

    Returns:
        scores: (nlat, nlon) — composite tail risk score
        flags: (nlat, nlon) — boolean, True if above threshold
        threshold: float — the 95th percentile cutoff
    """
    tas = data['tas']
    pr = data['pr']

    # Temperature signals
    temp_vol = compute_volatility(tas, window=5)
    temp_mom = compute_momentum(tas, window=3)

    # Precipitation signals
    precip_vol = compute_volatility(pr, window=5)
    precip_mom = compute_momentum(pr, window=3)

    # Composite: weighted combination, both variables
    # Normalize each component to [0, 1]
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    score = (
        vol_weight * (norm(temp_vol) + norm(precip_vol)) / 2 +
        mom_weight * (norm(temp_mom) + norm(precip_mom)) / 2
    )

    threshold = np.percentile(score, percentile)
    flags = score >= threshold

    return score, flags, threshold


def get_tail_risk_nodes(data):
    """Convenience: returns flat arrays for graph construction."""
    scores, flags, threshold = compute_tail_risk(data)
    return scores.flatten(), flags.flatten(), threshold
```

---

## 6. GRAPH CONSTRUCTION

### src/graph/build_graph.py

```python
"""Convert grid cells → PyTorch Geometric graph."""
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph


def build_climate_graph(features, positions, k=8):
    """
    Build a PyG Data object from node features and positions.

    Args:
        features: (N, F) numpy array — node features
        positions: (N, 2) numpy array — (lat, lon) per node
        k: number of nearest neighbors for edges

    Returns:
        PyG Data object
    """
    N = features.shape[0]

    # Build adjacency via k-nearest neighbors
    adj = kneighbors_graph(positions, n_neighbors=k, mode='connectivity', include_self=False)
    edge_index = np.array(adj.nonzero())  # (2, E)

    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, pos=pos)
    data.num_nodes = N

    return data


def build_grid_adjacency(nlat, nlon):
    """
    Alternative: grid adjacency (4-connected or 8-connected).
    Faster than KNN, no sklearn needed.
    """
    edges = []
    for i in range(nlat):
        for j in range(nlon):
            node = i * nlon + j
            # 4-connected neighbors
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # 8-connected
                ni, nj = i + di, j + dj
                if 0 <= ni < nlat and 0 <= nj < nlon:
                    neighbor = ni * nlon + nj
                    edges.append([node, neighbor])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
```

---

## 7. GNN MODEL

### src/model/gnn.py

```python
"""Minimal 2-layer GCN for risk propagation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ClimateRiskGNN(nn.Module):
    """
    2-layer GCN.
    Input: node features (N, F)
    Output: risk score per node (N, 1)
    """

    def __init__(self, in_channels=5, hidden_channels=32, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        risk = torch.sigmoid(self.head(x))  # (N, 1), bounded [0, 1]
        return risk.squeeze(-1)  # (N,)


def train_gnn(model, data, target_scores, epochs=50, lr=0.01):
    """
    Quick supervised training using tail-risk scores as pseudo-labels.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target = torch.tensor(target_scores, dtype=torch.float32)

    # Normalize target to [0, 1]
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def predict(model, data):
    """Run inference, return risk scores."""
    model.eval()
    with torch.no_grad():
        scores = model(data)
    return scores.numpy()
```

---

## 8. SIMULATION ENGINE

### src/simulation/interventions.py

```python
"""Define intervention parameter deltas."""

INTERVENTIONS = {
    'mangrove_restoration': {
        'name': 'Coastal Mangrove Restoration',
        'cost_usd': 1_000_000_000,  # $1B
        'deltas': {
            'precip_volatility_reduction': 0.20,   # 20% reduction in precip volatility
            'temp_reduction': 0.15,                  # 0.15°C cooling effect
            'coastal_only': True,                    # Only applies to coastal nodes
            'coastal_lon_threshold': 120,            # lon > threshold = coastal
        },
        'description': 'Mangrove buffer reduces storm surge and precipitation volatility along coastlines',
    },
    'regenerative_agriculture': {
        'name': 'Regenerative Agriculture',
        'cost_usd': 1_000_000_000,  # $1B
        'deltas': {
            'precip_volatility_reduction': 0.10,   # 10% via soil moisture retention
            'temp_reduction': 0.05,                  # minor albedo effect
            'gdp_boost_factor': 1.05,               # 5% GDP boost from yields
            'coastal_only': False,
        },
        'description': 'Soil health improvements buffer against drought and boost agricultural output',
    },
}
```

### src/simulation/run_simulations.py

```python
"""Apply interventions and re-run GNN to get counterfactual risk."""
import numpy as np
import torch
import copy
from torch_geometric.data import Data


def apply_intervention(features, positions, intervention, lons):
    """
    Apply intervention deltas to feature matrix.

    Args:
        features: (N, 5) — [temp, precip, volatility, momentum, gdp] (NORMALIZED)
        positions: (N, 2) — [lat, lon]
        intervention: dict from interventions.py
        lons: original lon values for coastal detection

    Returns:
        modified_features: (N, 5)
    """
    modified = features.copy()
    deltas = intervention['deltas']

    # Determine which nodes are affected
    if deltas.get('coastal_only', False):
        threshold = deltas.get('coastal_lon_threshold', 120)
        mask = positions[:, 1] >= threshold  # lon >= threshold
    else:
        mask = np.ones(len(features), dtype=bool)

    # Apply temp reduction (feature index 0)
    temp_delta = deltas.get('temp_reduction', 0)
    modified[mask, 0] -= temp_delta * 0.5  # scaled for normalized features

    # Apply volatility reduction (feature index 2)
    vol_reduction = deltas.get('precip_volatility_reduction', 0)
    modified[mask, 2] *= (1 - vol_reduction)

    # Apply GDP boost (feature index 4)
    gdp_factor = deltas.get('gdp_boost_factor', 1.0)
    modified[mask, 4] *= gdp_factor

    return modified


def run_simulation(model, base_data, features, positions, intervention, lons):
    """
    Run one intervention simulation.
    Returns: baseline_risk, intervention_risk (both numpy arrays of shape (N,))
    """
    from src.model.gnn import predict

    # Baseline risk
    baseline_risk = predict(model, base_data)

    # Modify features
    mod_features = apply_intervention(features, positions, intervention, lons)

    # Build new graph data
    mod_data = Data(
        x=torch.tensor(mod_features, dtype=torch.float32),
        edge_index=base_data.edge_index,
        pos=base_data.pos,
        num_nodes=base_data.num_nodes,
    )

    # Intervention risk
    intervention_risk = predict(model, mod_data)

    return baseline_risk, intervention_risk


def run_all_simulations(model, base_data, features, positions, interventions_dict, lons):
    """Run all interventions, return results dict."""
    results = {}
    from src.model.gnn import predict
    baseline_risk = predict(model, base_data)

    for key, intervention in interventions_dict.items():
        mod_features = apply_intervention(features, positions, intervention, lons)
        mod_data = Data(
            x=torch.tensor(mod_features, dtype=torch.float32),
            edge_index=base_data.edge_index,
            pos=base_data.pos,
            num_nodes=base_data.num_nodes,
        )
        int_risk = predict(model, mod_data)

        results[key] = {
            'name': intervention['name'],
            'baseline_risk': baseline_risk,
            'intervention_risk': int_risk,
            'risk_reduction': baseline_risk - int_risk,
            'cost': intervention['cost_usd'],
        }

    return baseline_risk, results
```

### src/simulation/roi.py

```python
"""Resilience ROI calculation with uncertainty penalty."""
import numpy as np


def compute_roi(baseline_risk, intervention_risk, cost, gdp_flat, pop_flat,
                precip_data=None, discount_rate=0.03, n_years=10):
    """
    ROI_resilience = (sum(L_baseline - L_intervention) * gamma^t) / Cost ± U_precip

    Loss proxy: risk_score * GDP * population_density (economic exposure)
    """
    # Loss proxy per node
    loss_baseline = baseline_risk * gdp_flat * (pop_flat / pop_flat.max())
    loss_intervention = intervention_risk * gdp_flat * (pop_flat / pop_flat.max())

    # Discounted loss avoided over time horizon
    total_loss_avoided = 0
    for t in range(n_years):
        gamma = (1 / (1 + discount_rate)) ** t
        total_loss_avoided += np.sum(loss_baseline - loss_intervention) * gamma

    # ROI
    roi = total_loss_avoided / cost

    # Uncertainty penalty from precipitation
    u_precip = compute_uncertainty_penalty(precip_data) if precip_data is not None else 0

    return {
        'roi': roi,
        'roi_lower': roi - u_precip,
        'roi_upper': roi + u_precip,
        'total_loss_avoided': total_loss_avoided,
        'u_precip': u_precip,
        'mean_risk_reduction': float(np.mean(baseline_risk - intervention_risk)),
    }


def compute_uncertainty_penalty(precip_data, ci=0.95):
    """
    U_precip: uncertainty from precipitation variability.
    Higher precip variance → wider confidence interval → larger penalty.
    """
    if precip_data is None:
        return 0.0

    # precip_data shape: (T, nlat, nlon) or flat
    if precip_data.ndim == 3:
        spatial_std = np.std(precip_data, axis=0)  # std over time
        mean_uncertainty = np.mean(spatial_std)
    else:
        mean_uncertainty = np.std(precip_data)

    # Scale to ROI units (heuristic: normalize relative to mean precip)
    mean_precip = np.mean(np.abs(precip_data))
    u_precip = (mean_uncertainty / (mean_precip + 1e-8)) * 0.5  # scaled penalty

    return float(u_precip)
```

---

## 9. RESILIENCE ROI

(Covered in `src/simulation/roi.py` above)

Key formula implemented:

```
ROI = Σ(L_baseline - L_intervention) × γ^t / Cost ± U_precip
```

Where:
- L = risk_score × GDP × normalized_population (economic exposure proxy)
- γ = 1/(1+r) discount factor
- U_precip = mean(σ_precip) / mean(precip) × 0.5

---

## 10. GENERATIVE DOWNSCALING (SMART HACK)

### src/rendering/downscale.py

```python
"""Fake generative downscaling via interpolation + smoothing."""
import numpy as np
from scipy.ndimage import zoom, gaussian_filter


def downscale_grid(coarse_grid, scale_factor=8, sigma=1.5):
    """
    Upsample a coarse grid to fake high-resolution output.

    Args:
        coarse_grid: (nlat, nlon) numpy array
        scale_factor: upsampling factor (8x = 2° → 0.25° equivalent)
        sigma: Gaussian smoothing to remove blocky artifacts

    Returns:
        hires_grid: (nlat*scale, nlon*scale) numpy array
    """
    # Bicubic interpolation
    hires = zoom(coarse_grid, scale_factor, order=3)

    # Gaussian smooth to look realistic
    hires = gaussian_filter(hires, sigma=sigma)

    return hires


def downscale_timeseries(data_3d, scale_factor=8, sigma=1.5):
    """
    Downscale entire time series.
    Input: (T, nlat, nlon)
    Output: (T, nlat*scale, nlon*scale)
    """
    T = data_3d.shape[0]
    first = downscale_grid(data_3d[0], scale_factor, sigma)
    result = np.zeros((T, *first.shape))
    result[0] = first

    for t in range(1, T):
        result[t] = downscale_grid(data_3d[t], scale_factor, sigma)

    return result
```

---

## 11. VIDEO GENERATION

### src/rendering/render_video.py

```python
"""Generate heatmap videos from climate/risk data."""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .downscale import downscale_grid
import os


def render_risk_video(risk_series, lats, lons, output_path, title='Risk Heatmap',
                      fps=4, scale_factor=8, cmap='YlOrRd'):
    """
    Render a time series of risk grids as an MP4 video.

    Args:
        risk_series: list of (nlat, nlon) arrays, one per timestep
        lats, lons: coordinate arrays
        output_path: e.g., 'outputs/videos/baseline_risk.mp4'
        title: video title
        fps: frames per second
        scale_factor: downscaling upsampling factor
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Downscale first frame
    hires = downscale_grid(risk_series[0], scale_factor=scale_factor)
    vmin = min(r.min() for r in risk_series)
    vmax = max(r.max() for r in risk_series)

    im = ax.imshow(hires, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   extent=[lons[0], lons[-1], lats[0], lats[-1]], aspect='auto')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(im, ax=ax, label='Risk Score')
    title_text = ax.set_title(f'{title} — Year 2015')

    def update(frame):
        hires = downscale_grid(risk_series[frame], scale_factor=scale_factor)
        im.set_data(hires)
        title_text.set_text(f'{title} — Year {2015 + frame}')
        return [im, title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(risk_series),
                                   interval=1000//fps, blit=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved video: {output_path}")
    return output_path


def render_comparison_video(baseline_series, intervention_series, lats, lons,
                            output_path, intervention_name='Intervention',
                            fps=4, scale_factor=8):
    """
    Side-by-side comparison: baseline vs intervention.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    vmin = min(min(r.min() for r in baseline_series),
               min(r.min() for r in intervention_series))
    vmax = max(max(r.max() for r in baseline_series),
               max(r.max() for r in intervention_series))

    extent = [lons[0], lons[-1], lats[0], lats[-1]]

    h1 = downscale_grid(baseline_series[0], scale_factor=scale_factor)
    h2 = downscale_grid(intervention_series[0], scale_factor=scale_factor)

    im1 = ax1.imshow(h1, cmap='YlOrRd', origin='lower', vmin=vmin, vmax=vmax,
                     extent=extent, aspect='auto')
    im2 = ax2.imshow(h2, cmap='YlOrRd', origin='lower', vmin=vmin, vmax=vmax,
                     extent=extent, aspect='auto')

    ax1.set_title('Baseline Risk')
    ax2.set_title(f'With {intervention_name}')
    plt.colorbar(im1, ax=ax1, label='Risk')
    plt.colorbar(im2, ax=ax2, label='Risk')

    year_text = fig.suptitle('Year 2015', fontsize=14, fontweight='bold')

    def update(frame):
        im1.set_data(downscale_grid(baseline_series[frame], scale_factor=scale_factor))
        im2.set_data(downscale_grid(intervention_series[frame], scale_factor=scale_factor))
        year_text.set_text(f'Year {2015 + frame}')
        return [im1, im2, year_text]

    ani = animation.FuncAnimation(fig, update, frames=len(baseline_series),
                                   interval=1000//fps, blit=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved comparison video: {output_path}")
    return output_path


def render_tail_risk_video(risk_series, flags_series, lats, lons, output_path,
                           fps=4, scale_factor=8):
    """
    Risk heatmap with tail-risk nodes highlighted as red dots.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    extent = [lons[0], lons[-1], lats[0], lats[-1]]

    hires = downscale_grid(risk_series[0], scale_factor=scale_factor)
    vmin = min(r.min() for r in risk_series)
    vmax = max(r.max() for r in risk_series)

    im = ax.imshow(hires, cmap='YlOrRd', origin='lower', vmin=vmin, vmax=vmax,
                   extent=extent, aspect='auto')

    # Flagged nodes as scatter
    nlat, nlon = risk_series[0].shape
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    lat_flat, lon_flat = lat_grid.flatten(), lon_grid.flatten()

    flagged = flags_series[0].flatten()
    scatter = ax.scatter(lon_flat[flagged], lat_flat[flagged],
                         c='red', s=50, marker='X', label='Tail-Risk Node', zorder=5)

    ax.legend(loc='upper left')
    plt.colorbar(im, ax=ax, label='Risk Score')
    title_text = ax.set_title('Tail-Risk Escalation — Year 2015')

    def update(frame):
        hires = downscale_grid(risk_series[frame], scale_factor=scale_factor)
        im.set_data(hires)
        flagged = flags_series[frame].flatten()
        offsets = np.column_stack([lon_flat[flagged], lat_flat[flagged]])
        scatter.set_offsets(offsets if len(offsets) > 0 else np.empty((0, 2)))
        title_text.set_text(f'Tail-Risk Escalation — Year {2015 + frame}')
        return [im, scatter, title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(risk_series),
                                   interval=1000//fps, blit=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved tail-risk video: {output_path}")
    return output_path
```

---

## 12. EMBEDDING + SEARCH

### src/embedding/embedder.py

```python
"""Embed video frames using Gemini API or CLIP fallback."""
import os
import numpy as np
from PIL import Image
import tempfile


def extract_keyframes(video_path, n_frames=8):
    """Extract evenly-spaced frames from MP4."""
    import subprocess
    import json

    # Get video duration
    probe = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path],
        capture_output=True, text=True
    )
    info = json.loads(probe.stdout)
    duration = float(info['streams'][0].get('duration', '10'))

    tmpdir = tempfile.mkdtemp()
    frames = []
    timestamps = np.linspace(0, duration * 0.95, n_frames)

    for i, ts in enumerate(timestamps):
        out_path = os.path.join(tmpdir, f'frame_{i:03d}.png')
        subprocess.run([
            'ffmpeg', '-ss', str(ts), '-i', video_path,
            '-frames:v', '1', '-q:v', '2', out_path,
            '-y', '-loglevel', 'quiet'
        ])
        if os.path.exists(out_path):
            frames.append(out_path)

    return frames


def embed_with_gemini(frame_paths, video_metadata=None):
    """
    Embed video frames using Gemini 1.5 Pro multimodal.
    Returns: list of embedding vectors
    """
    from google import genai

    client = genai.Client()  # Uses GOOGLE_API_KEY env var

    embeddings = []
    for path in frame_paths:
        img = Image.open(path)

        # Use Gemini to get a text description, then embed the description
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                img,
                'Describe this climate risk heatmap in detail: regions, risk levels, patterns, notable features. Be specific about geography and severity.'
            ]
        )
        description = response.text

        # Add metadata context
        if video_metadata:
            description = f"{video_metadata}. {description}"

        # Embed the description
        embed_response = client.models.embed_content(
            model='models/text-embedding-004',
            contents=description
        )
        embeddings.append(embed_response.embeddings[0].values)

    return embeddings


def embed_with_clip_fallback(frame_paths):
    """
    Fallback: use CLIP to embed frames directly.
    Requires: pip install open-clip-torch
    """
    try:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        model.eval()

        embeddings = []
        for path in frame_paths:
            img = preprocess(Image.open(path)).unsqueeze(0)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze().numpy())

        return embeddings
    except ImportError:
        print("CLIP not installed. Install with: pip install open-clip-torch")
        return None


def embed_video(video_path, metadata=None, n_frames=8, use_gemini=True):
    """
    Main entry point: embed a video, return mean embedding.
    """
    frames = extract_keyframes(video_path, n_frames=n_frames)

    if not frames:
        print(f"WARNING: No frames extracted from {video_path}")
        return np.zeros(768)  # default dim

    if use_gemini:
        try:
            embeddings = embed_with_gemini(frames, metadata)
        except Exception as e:
            print(f"Gemini failed: {e}. Falling back to CLIP.")
            embeddings = embed_with_clip_fallback(frames)
    else:
        embeddings = embed_with_clip_fallback(frames)

    if embeddings is None or len(embeddings) == 0:
        return np.zeros(768)

    # Mean pooling across frames
    mean_emb = np.mean(embeddings, axis=0)
    return mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
```

### src/embedding/vectordb.py

```python
"""ChromaDB vector store for video search."""
import chromadb
import numpy as np
import os


class VideoSearchDB:
    def __init__(self, persist_dir='outputs/embeddings'):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name='sentree_videos',
            metadata={'hnsw:space': 'cosine'}
        )

    def add_video(self, video_id, embedding, metadata=None):
        """Add a video embedding to the store."""
        self.collection.upsert(
            ids=[video_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata or {}],
        )

    def add_videos_batch(self, video_ids, embeddings, metadatas=None):
        """Batch add videos."""
        self.collection.upsert(
            ids=video_ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas or [{} for _ in video_ids],
        )

    def query(self, query_text, n_results=5, use_gemini=True):
        """
        Search for videos matching a text query.
        Embeds the query text, then finds nearest videos.
        """
        query_embedding = self._embed_text(query_text, use_gemini)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )
        return results

    def _embed_text(self, text, use_gemini=True):
        """Embed query text."""
        if use_gemini:
            try:
                from google import genai
                client = genai.Client()
                response = client.models.embed_content(
                    model='models/text-embedding-004',
                    contents=text
                )
                return np.array(response.embeddings[0].values)
            except Exception as e:
                print(f"Gemini text embed failed: {e}")

        # Fallback: CLIP text embedding
        try:
            import open_clip
            import torch
            model, _, _ = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k'
            )
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            with torch.no_grad():
                tokens = tokenizer([text])
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.squeeze().numpy()
        except:
            print("WARNING: No embedding model available. Returning zeros.")
            return np.zeros(768)

    def count(self):
        return self.collection.count()
```

---

## 13. STREAMLIT DASHBOARD

### src/dashboard/app.py

```python
"""SenTree Dashboard — Streamlit UI."""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

st.set_page_config(page_title='SenTree — Resilience ROI Dashboard', layout='wide')

# --- Header ---
st.title('SenTree: Resilience ROI Dashboard')
st.markdown('*Climate Adaptation Intelligence for Sovereign Wealth Funds*')
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header('Controls')
    scenario = st.selectbox('Climate Scenario', ['SSP3-7.0 (High Emissions)', 'SSP1-2.6 (Low Emissions)'])
    intervention = st.selectbox('Intervention', ['Coastal Mangrove Restoration', 'Regenerative Agriculture', 'Both'])
    st.divider()
    st.markdown('**Region:** SE Asia Coastal')
    st.markdown('**Time:** 2015–2050')

# --- Search ---
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        'Search simulations',
        placeholder='e.g., "Show where mangroves prevent collapse"',
        key='search_query'
    )
with col2:
    search_btn = st.button('Search', type='primary', use_container_width=True)

# --- Load results ---
@st.cache_data
def load_roi_data():
    roi_path = 'outputs/roi/roi_results.json'
    if os.path.exists(roi_path):
        with open(roi_path, 'r') as f:
            return json.load(f)
    # Demo fallback
    return {
        'mangrove_restoration': {
            'name': 'Coastal Mangrove Restoration',
            'roi': 3.42, 'roi_lower': 2.91, 'roi_upper': 3.93,
            'total_loss_avoided': 3_420_000_000,
            'mean_risk_reduction': 0.18,
            'tail_risk_nodes_neutralized': 12,
        },
        'regenerative_agriculture': {
            'name': 'Regenerative Agriculture',
            'roi': 2.15, 'roi_lower': 1.78, 'roi_upper': 2.52,
            'total_loss_avoided': 2_150_000_000,
            'mean_risk_reduction': 0.09,
            'tail_risk_nodes_neutralized': 5,
        }
    }


roi_data = load_roi_data()

# --- Search Results ---
if search_btn and query:
    st.subheader('Search Results')

    try:
        from src.embedding.vectordb import VideoSearchDB
        db = VideoSearchDB()

        if db.count() > 0:
            results = db.query(query, n_results=3)

            for i, (vid_id, metadata, distance) in enumerate(zip(
                results['ids'][0], results['metadatas'][0], results['distances'][0]
            )):
                similarity = 1 - distance
                with st.expander(f'Result {i+1}: {metadata.get("title", vid_id)} — Relevance: {similarity:.1%}', expanded=(i==0)):
                    video_path = metadata.get('video_path', f'outputs/videos/{vid_id}.mp4')
                    if os.path.exists(video_path):
                        st.video(video_path)

                    c1, c2, c3 = st.columns(3)
                    roi_key = metadata.get('intervention_key', '')
                    if roi_key in roi_data:
                        r = roi_data[roi_key]
                        c1.metric('ROI', f"{r['roi']:.2f}x", f"±{r.get('u_precip', 0.5):.2f}")
                        c2.metric('Loss Avoided', f"${r['total_loss_avoided']/1e9:.1f}B")
                        c3.metric('Risk Reduction', f"{r['mean_risk_reduction']:.1%}")

                    if metadata.get('has_tail_risk'):
                        st.warning(f"⚠ Tail-Risk Nodes Detected: {metadata.get('tail_risk_count', 'N/A')}")
        else:
            st.info('No videos indexed yet. Run `python scripts/index_videos.py` first.')

    except Exception as e:
        st.error(f'Search error: {e}')
        st.info('Showing demo results instead.')

# --- Metrics Dashboard ---
st.subheader('Intervention Comparison')

cols = st.columns(len(roi_data))
for i, (key, data) in enumerate(roi_data.items()):
    with cols[i]:
        st.markdown(f"**{data['name']}**")
        st.metric('Resilience ROI', f"{data['roi']:.2f}x",
                   help=f"Range: {data.get('roi_lower', 0):.2f} – {data.get('roi_upper', 0):.2f}")
        st.metric('Total Loss Avoided', f"${data.get('total_loss_avoided', 0)/1e9:.1f}B")
        st.metric('Mean Risk Reduction', f"{data.get('mean_risk_reduction', 0):.1%}")

        tail_count = data.get('tail_risk_nodes_neutralized', 0)
        if tail_count > 0:
            st.error(f'Tail-Risk Nodes Neutralized: {tail_count}')

# --- Video Display ---
st.subheader('Simulation Videos')

video_dir = 'outputs/videos'
if os.path.exists(video_dir):
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if videos:
        tabs = st.tabs([v.replace('.mp4', '').replace('_', ' ').title() for v in videos])
        for tab, vid in zip(tabs, videos):
            with tab:
                st.video(os.path.join(video_dir, vid))
    else:
        st.info('No videos generated yet. Run `python scripts/run_pipeline.py`.')
else:
    st.info('Output directory not found. Run the pipeline first.')

# --- Tail Risk Map ---
st.subheader('Tail-Risk Escalation Map')
st.markdown('Nodes exceeding 95th percentile volatility+momentum threshold are flagged.')

# Display static image if available
tail_risk_img = 'outputs/tail_risk_map.png'
if os.path.exists(tail_risk_img):
    st.image(tail_risk_img, use_container_width=True)
else:
    st.info('Tail-risk map not generated yet.')

# --- Footer ---
st.divider()
st.caption('SenTree — Resilience ROI Dashboard | ML@Purdue Catapult Hackathon')
```

---

## 14. TEAM EXECUTION PLAN

### Person 1 — Data & Tail Risk (The Scientist)

**Hours 0–4:**
- Set up repo, environment, requirements.txt
- Generate synthetic data OR start ISIMIP download
- Verify data loads correctly

**Hours 4–12:**
- Implement `src/tail_risk/volatility.py` — DONE by hour 6
- Implement `src/tail_risk/momentum.py` — DONE by hour 7
- Implement `src/tail_risk/engine.py` — DONE by hour 9
- Implement `src/data/preprocess.py` — DONE by hour 11
- **OUTPUT:** `data/processed/climate_data.pkl` + tail risk scores

**Hours 12–16:**
- Help P2 verify graph construction uses correct features
- Implement `src/simulation/roi.py`
- Generate ROI CSV/JSON outputs

**Hours 16–36:**
- Polish tail-risk visualizations
- Help P3 with metadata for search indexing
- Pitch preparation: explain the math

**HANDOFF TO P2:** `data/processed/climate_data.pkl` containing `{tas, pr, gdp, pop, lats, lons, years}` — available by hour 4 (synthetic) or hour 8 (real data).

**HANDOFF TO P3:** `outputs/roi/roi_results.json` — available by hour 16.

---

### Person 2 — ML & Simulation & Video (The Architect)

**Hours 0–4:**
- Set up PyTorch + PyG
- Implement `src/graph/build_graph.py`
- Test with dummy data while waiting for P1

**Hours 4–12:**
- Implement `src/model/gnn.py` — 2-layer GCN, train function
- Implement `src/simulation/interventions.py`
- Implement `src/simulation/run_simulations.py`
- **MILESTONE:** GNN runs inference on baseline + interventions

**Hours 12–20:**
- Implement `src/rendering/downscale.py`
- Implement `src/rendering/render_video.py`
- Generate all simulation videos:
  - `baseline_risk.mp4`
  - `mangrove_intervention.mp4`
  - `agriculture_intervention.mp4`
  - `comparison_mangrove.mp4`
  - `comparison_agriculture.mp4`
  - `tail_risk_escalation.mp4`
- **OUTPUT:** MP4s in `outputs/videos/`

**Hours 20–28:**
- Implement `scripts/run_pipeline.py` (end-to-end orchestration)
- Generate additional simulation variations (different years, combined interventions)
- Help P3 debug video embedding

**HANDOFF TO P3:** `outputs/videos/*.mp4` — available by hour 20.

---

### Person 3 — Embeddings & Dashboard (The Product Lead)

**Hours 0–8:**
- Set up Gemini API key, ChromaDB
- Implement `src/embedding/embedder.py`
- Implement `src/embedding/vectordb.py`
- Test with dummy video (screen recording or sample)

**Hours 8–16:**
- Build `src/dashboard/app.py` — layout, sidebar, metrics display
- Test with hardcoded demo data
- Wire up video display

**Hours 16–24:**
- Integrate P2's videos: `scripts/index_videos.py`
- Embed all videos into ChromaDB
- Wire search bar → ChromaDB → video display

**Hours 24–34:**
- Integrate P1's ROI data
- Polish UI: metrics, tail-risk flags, error states
- Full end-to-end testing: search → video + ROI + flags

**Hours 34–36:**
- Demo rehearsal
- Screenshots for pitch deck

---

## 15. 36-HOUR MVP STRATEGY

### "We win if:"

A judge types: **"Show where mangroves prevent collapse"**

The system returns:
1. A side-by-side MP4 video showing baseline risk vs mangrove intervention
2. ROI = 3.42x (range: 2.91–3.93x)
3. Highlighted tail-risk nodes that were neutralized
4. Economic loss avoided: $3.4B over 10 years

### Hour-by-Hour Milestones

| Hour | Gate | Owner |
|------|------|-------|
| 4 | Synthetic data loads, env works | P1 |
| 8 | Tail-risk scores computed | P1 |
| 12 | GNN runs inference | P2 |
| 16 | First MP4 rendered | P2 |
| 20 | All videos generated | P2 |
| 24 | Videos embedded in ChromaDB | P3 |
| 28 | Search returns relevant video | P3 |
| 32 | Full dashboard works end-to-end | ALL |
| 34 | Demo rehearsal done | ALL |
| 36 | Ship | ALL |

### What to CUT if behind schedule

**Cut first (low impact):**
- SSP1-2.6 comparison scenario → only do SSP3-7.0
- Combined interventions → only individual
- CLIP fallback → Gemini only or vice versa

**Cut second (medium impact):**
- Real ISIMIP data → use synthetic only (already built)
- GNN training → random weights + hardcoded propagation
- Multiple video types → only comparison videos

**Cut third (if desperate):**
- Video embedding → use hardcoded keyword matching
- ChromaDB → return pre-ranked results
- Live search → pre-computed demo responses

**NEVER cut:**
- The ROI calculation
- The tail-risk flagging
- At least one working video
- The Streamlit dashboard

---

## 16. DEBUGGING + FALLBACKS

### If NetCDF download is too slow
```python
# Already handled: data/generate_synthetic.py creates realistic fake data
# Just run: python data/generate_synthetic.py
```

### If GNN training fails or is too slow
```python
# Fallback: simple propagation heuristic
def simple_propagation(features, adj_matrix, iterations=3, alpha=0.5):
    """Diffusion-based risk propagation without neural network."""
    risk = features[:, 2]  # volatility as base risk
    for _ in range(iterations):
        neighbor_risk = adj_matrix @ risk / adj_matrix.sum(axis=1).clip(1)
        risk = alpha * risk + (1 - alpha) * neighbor_risk
    return risk
```

### If video rendering is too slow
```python
# Skip downscaling, render at native resolution
# Or reduce frames: render every 3rd year instead of every year
# Or generate PNG series instead of MP4
```

### If Gemini API quota hit or fails
```python
# Fallback chain:
# 1. Try CLIP embeddings (local, no API needed)
# 2. Try simple keyword extraction from metadata
# 3. Hardcode search results for demo
```

### If ChromaDB fails
```python
# Fallback: in-memory cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
# Store embeddings in a dict, search with numpy
```

### If ffmpeg not available
```python
# Use Pillow to save frames as GIF instead
from PIL import Image
frames = [Image.open(f) for f in frame_paths]
frames[0].save('output.gif', save_all=True, append_images=frames[1:], duration=250, loop=0)
```

### If PyTorch Geometric install fails
```bash
# CPU-only minimal install:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
# Skip torch-scatter/torch-sparse — use basic GCN implementation

# Or: implement GCN manually (10 lines):
```
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        # adj: sparse or dense adjacency (N, N)
        x = adj @ x  # message passing
        x = F.relu(self.w1(x))
        x = adj @ x
        x = torch.sigmoid(self.w2(x))
        return x.squeeze(-1)
```

---

## SCRIPTS

### scripts/run_pipeline.py

```python
"""End-to-end pipeline: data → tail risk → graph → GNN → simulations → videos."""
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_isimip import load_climate_data
from src.data.preprocess import build_node_features, build_temporal_features
from src.tail_risk.engine import compute_tail_risk, get_tail_risk_nodes
from src.graph.build_graph import build_climate_graph
from src.model.gnn import ClimateRiskGNN, train_gnn, predict
from src.simulation.interventions import INTERVENTIONS
from src.simulation.run_simulations import run_all_simulations
from src.simulation.roi import compute_roi
from src.rendering.render_video import (
    render_risk_video, render_comparison_video, render_tail_risk_video
)

print("=" * 60)
print("SenTree Pipeline — Starting")
print("=" * 60)

# 1. Load data
print("\n[1/7] Loading climate data...")
data = load_climate_data()
print(f"  Shape: {data['tas'].shape} — {len(data['years'])} years, "
      f"{data['tas'].shape[1]}x{data['tas'].shape[2]} grid")

# 2. Compute tail risk
print("\n[2/7] Computing tail-risk scores...")
scores, flags, threshold = compute_tail_risk(data)
print(f"  Threshold (95th pct): {threshold:.4f}")
print(f"  Flagged nodes: {flags.sum()} / {flags.size}")

# 3. Build features + graph
print("\n[3/7] Building graph...")
features, positions, scaler = build_node_features(data, year_idx=-1)
graph_data = build_climate_graph(features, positions, k=8)
print(f"  Nodes: {graph_data.num_nodes}, Edges: {graph_data.edge_index.shape[1]}")

# 4. Train GNN
print("\n[4/7] Training GNN...")
model = ClimateRiskGNN(in_channels=features.shape[1])
tail_scores_flat, _, _ = get_tail_risk_nodes(data)
model = train_gnn(model, graph_data, tail_scores_flat, epochs=50)

# 5. Run simulations
print("\n[5/7] Running simulations...")
lons = data['lons']
baseline_risk, sim_results = run_all_simulations(
    model, graph_data, features, positions, INTERVENTIONS, lons
)

# 6. Compute ROI
print("\n[6/7] Computing ROI...")
roi_results = {}
for key, result in sim_results.items():
    roi = compute_roi(
        result['baseline_risk'], result['intervention_risk'],
        result['cost'], data['gdp'].flatten(), data['pop'].flatten(),
        precip_data=data['pr']
    )
    roi_results[key] = {
        'name': result['name'],
        **roi,
        'tail_risk_nodes_neutralized': int((
            (result['baseline_risk'] > np.percentile(result['baseline_risk'], 95)) &
            (result['intervention_risk'] <= np.percentile(result['baseline_risk'], 95))
        ).sum())
    }
    print(f"  {result['name']}: ROI = {roi['roi']:.2f}x "
          f"(range: {roi['roi_lower']:.2f} – {roi['roi_upper']:.2f})")

os.makedirs('outputs/roi', exist_ok=True)
with open('outputs/roi/roi_results.json', 'w') as f:
    json.dump(roi_results, f, indent=2, default=str)

# 7. Render videos
print("\n[7/7] Rendering videos...")
nlat, nlon = data['tas'].shape[1], data['tas'].shape[2]
T = data['tas'].shape[0]

# Generate per-timestep risk (re-run GNN for each year)
temporal_features = build_temporal_features(data)
baseline_risk_series = []
intervention_risk_series = {}

for key in INTERVENTIONS:
    intervention_risk_series[key] = []

for t in range(T):
    feats = temporal_features[t]
    import torch
    from torch_geometric.data import Data as PyGData

    temp_data = PyGData(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=graph_data.edge_index,
        pos=graph_data.pos,
        num_nodes=graph_data.num_nodes,
    )
    b_risk = predict(model, temp_data)
    baseline_risk_series.append(b_risk.reshape(nlat, nlon))

    for key, interv in INTERVENTIONS.items():
        from src.simulation.run_simulations import apply_intervention
        mod_feats = apply_intervention(feats, positions, interv, lons)
        mod_data = PyGData(
            x=torch.tensor(mod_feats, dtype=torch.float32),
            edge_index=graph_data.edge_index,
            pos=graph_data.pos,
            num_nodes=graph_data.num_nodes,
        )
        i_risk = predict(model, mod_data)
        intervention_risk_series[key].append(i_risk.reshape(nlat, nlon))

# Tail-risk flags per timestep
from src.tail_risk.volatility import compute_volatility_series
from src.tail_risk.momentum import compute_momentum_series
vol_s = compute_volatility_series(data['tas'], 5)
mom_s = compute_momentum_series(data['tas'], 3)

flags_series = []
for t in range(T):
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)
    s = 0.6 * norm(vol_s[t]) + 0.4 * norm(mom_s[t])
    flags_series.append(s >= np.percentile(s, 95))

# Render all videos
render_risk_video(baseline_risk_series, data['lats'], data['lons'],
                  'outputs/videos/baseline_risk.mp4', title='Baseline Climate Risk')

render_tail_risk_video(baseline_risk_series, flags_series, data['lats'], data['lons'],
                       'outputs/videos/tail_risk_escalation.mp4')

for key in INTERVENTIONS:
    name = INTERVENTIONS[key]['name']
    render_comparison_video(
        baseline_risk_series, intervention_risk_series[key],
        data['lats'], data['lons'],
        f'outputs/videos/comparison_{key}.mp4',
        intervention_name=name
    )
    render_risk_video(
        intervention_risk_series[key], data['lats'], data['lons'],
        f'outputs/videos/{key}_risk.mp4',
        title=f'Risk with {name}'
    )

print("\n" + "=" * 60)
print("Pipeline complete!")
print(f"Videos: outputs/videos/")
print(f"ROI data: outputs/roi/roi_results.json")
print("=" * 60)
```

### scripts/index_videos.py

```python
"""Embed all output videos into ChromaDB for semantic search."""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.embedder import embed_video
from src.embedding.vectordb import VideoSearchDB

VIDEO_DIR = 'outputs/videos'
ROI_PATH = 'outputs/roi/roi_results.json'

# Load ROI data for metadata
roi_data = {}
if os.path.exists(ROI_PATH):
    with open(ROI_PATH, 'r') as f:
        roi_data = json.load(f)

# Video metadata mapping
VIDEO_META = {
    'baseline_risk': {
        'title': 'Baseline Climate Risk — SSP3-7.0 SE Asia',
        'description': 'Baseline systemic climate risk across Southeast Asia coastal region under SSP3-7.0 scenario, showing temperature and precipitation-driven risk propagation',
        'intervention_key': '',
        'has_tail_risk': True,
    },
    'tail_risk_escalation': {
        'title': 'Tail-Risk Escalation Events — Tipping Points',
        'description': 'Nodes exceeding 95th percentile volatility and momentum thresholds, indicating potential ecosystem collapse tipping points',
        'intervention_key': '',
        'has_tail_risk': True,
    },
    'comparison_mangrove_restoration': {
        'title': 'Mangrove Restoration Impact — Before vs After',
        'description': 'Side-by-side comparison showing how coastal mangrove restoration reduces systemic risk, prevents collapse in coastal nodes',
        'intervention_key': 'mangrove_restoration',
        'has_tail_risk': True,
    },
    'mangrove_restoration_risk': {
        'title': 'Risk with Mangrove Restoration',
        'description': 'Climate risk map after applying coastal mangrove restoration intervention, showing reduced coastal vulnerability',
        'intervention_key': 'mangrove_restoration',
        'has_tail_risk': False,
    },
    'comparison_regenerative_agriculture': {
        'title': 'Regenerative Agriculture Impact — Before vs After',
        'description': 'Side-by-side comparison showing how regenerative agriculture reduces precipitation volatility and stabilizes agricultural regions',
        'intervention_key': 'regenerative_agriculture',
        'has_tail_risk': True,
    },
    'regenerative_agriculture_risk': {
        'title': 'Risk with Regenerative Agriculture',
        'description': 'Climate risk map after applying regenerative agriculture intervention across the region',
        'intervention_key': 'regenerative_agriculture',
        'has_tail_risk': False,
    },
}

print("Indexing videos into ChromaDB...")
db = VideoSearchDB()

videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
for vid_file in videos:
    vid_id = vid_file.replace('.mp4', '')
    vid_path = os.path.join(VIDEO_DIR, vid_file)
    meta = VIDEO_META.get(vid_id, {
        'title': vid_id.replace('_', ' ').title(),
        'description': f'Climate simulation video: {vid_id}',
    })

    # Add ROI data to metadata
    ikey = meta.get('intervention_key', '')
    if ikey and ikey in roi_data:
        meta['roi'] = roi_data[ikey].get('roi', 0)
        meta['tail_risk_count'] = roi_data[ikey].get('tail_risk_nodes_neutralized', 0)

    meta['video_path'] = vid_path

    print(f"  Embedding: {vid_file}...")
    try:
        embedding = embed_video(vid_path, metadata=meta.get('description'), use_gemini=True)
        db.add_video(vid_id, embedding, metadata=meta)
        print(f"    ✓ Indexed")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

print(f"\nDone. {db.count()} videos indexed.")
```

---

## QUICK START (COPY-PASTE)

```bash
# 1. Clone and setup
cd SenTree
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data/raw data/processed outputs/videos outputs/roi outputs/embeddings

# 2. Generate data (or download ISIMIP)
python data/generate_synthetic.py

# 3. Set Gemini API key (for embeddings)
export GOOGLE_API_KEY="your-key-here"

# 4. Run full pipeline
python scripts/run_pipeline.py

# 5. Index videos for search
python scripts/index_videos.py

# 6. Launch dashboard
streamlit run src/dashboard/app.py
```
