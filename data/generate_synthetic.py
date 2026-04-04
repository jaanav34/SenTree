"""Generate synthetic ISIMIP-like climate data for hackathon use."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

import numpy as np
import pickle

np.random.seed(42)

# Grid: SE Asia coastal, 2-degree resolution
lats = np.arange(-10, 26, 2)   # 18 points
lons = np.arange(90, 132, 2)   # 21 points
years = np.arange(2015, 2051)   # 36 years
n_lats, n_lons, n_years = len(lats), len(lons), len(years)

# Temperature: base ~28C + warming trend + noise
temp_base = 28.0
temp_trend = np.linspace(0, 2.5, n_years)  # +2.5C by 2050 under SSP3-7.0
tas = np.zeros((n_years, n_lats, n_lons))
for t in range(n_years):
    tas[t] = temp_base + temp_trend[t] + np.random.normal(0, 0.5, (n_lats, n_lons))
    tas[t, :, -5:] += 0.3  # coastal cells warmer

# Precipitation: base ~5.5mm/day + increasing variability
pr_base = 5.5
pr = np.zeros((n_years, n_lats, n_lons))
for t in range(n_years):
    volatility_scale = 1.0 + 0.02 * t
    pr[t] = pr_base + np.random.normal(0, 1.5 * volatility_scale, (n_lats, n_lons))
    pr[t] = np.clip(pr[t], 0.1, 20)

# GDP per grid cell (proxy)
gdp = np.random.uniform(5000, 40000, (n_lats, n_lons))
gdp[:, -5:] *= 1.5  # coastal = richer

# Population density
pop = np.random.uniform(50, 5000, (n_lats, n_lons))
pop[:, -5:] *= 2  # coastal = denser

data = {
    'tas': tas,
    'pr': pr,
    'gdp': gdp,
    'pop': pop,
    'lats': lats,
    'lons': lons,
    'years': years,
}

out_dir = 'data/processed'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/climate_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"Generated synthetic data:")
print(f"  Years: {n_years} ({years[0]}-{years[-1]})")
print(f"  Grid: {n_lats} x {n_lons} = {n_lats * n_lons} nodes")
print(f"  Temp range: {tas.min():.1f} - {tas.max():.1f} C")
print(f"  Precip range: {pr.min():.1f} - {pr.max():.1f} mm/day")
print(f"  Saved to: {out_dir}/climate_data.pkl")
