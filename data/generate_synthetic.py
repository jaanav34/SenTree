"""
Generate high-fidelity synthetic ISIMIP-like climate data.

Matches real ISIMIP3b structure exactly:
  - Variable: tas (Near-Surface Air Temperature, °C after conversion from K)
  - Dims: (time, lat, lon) — stored as annual means (resampled from daily)
  - Resolution: 0.5° for SE Asia subset
  - Lat: ascending (-10 to 25)
  - Lon: 90 to 130
  - Time: 2015-2100 (86 years) — full SSP3-7.0 projection period

Also generates: pr (precipitation), GDP, population, soil_moisture, coastal_factor.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

import numpy as np
import pickle


def generate_synthetic_data(out_dir='data/processed'):
    """Generate and save synthetic data. Returns the data dict."""
    np.random.seed(42)

    # --- Grid: SE Asia coastal, 0.5-degree resolution ---
    # Matches real ISIMIP3b after subsetting
    lats = np.arange(-10, 25.5, 0.5)   # 71 points (ascending, like real data after flip)
    lons = np.arange(90, 130.5, 0.5)   # 81 points
    years = np.arange(2015, 2101)       # 86 years: 2015-2100 inclusive
    n_lats, n_lons, n_years = len(lats), len(lons), len(years)
    N = n_lats * n_lons

    print(f"Generating synthetic data: {n_years} years ({years[0]}-{years[-1]}), "
          f"{n_lats}x{n_lons} = {N} nodes")

    from scipy.ndimage import gaussian_filter

    # --- Spatial fields ---
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    # Temperature base (latitude-dependent, tropics warmer)
    temp_base_field = 30.0 - 0.15 * np.abs(lat_grid)

    # Coastal proximity
    coastal_distance = (lon_grid.max() - lon_grid) / (lon_grid.max() - lon_grid.min())
    coastal_factor = np.exp(-3 * coastal_distance)

    # Elevation proxy
    elevation_proxy = 0.3 * (1 - coastal_factor) * np.random.uniform(0, 1, (n_lats, n_lons))

    # --- Temperature (tas) in °C ---
    # SSP3-7.0: ~+2.5°C by 2050, ~+4.5°C by 2100 (non-linear)
    # Total range covers 86 years (2015-2100)
    year_frac = (years - 2015) / 85.0  # 0 to 1
    temp_trend = 4.5 * year_frac ** 1.3  # convex curve, ~2.5C at 2050, ~4.5C at 2100

    # Natural oscillations
    enso_cycle = 0.4 * np.sin(2 * np.pi * (years - 2015) / 5.2) + \
                 0.2 * np.sin(2 * np.pi * (years - 2015) / 3.7)
    ipo_cycle = 0.15 * np.sin(2 * np.pi * (years - 2015) / 20)

    tas = np.zeros((n_years, n_lats, n_lons), dtype=np.float32)
    for t in range(n_years):
        tas[t] = (
            temp_base_field +
            temp_trend[t] * (1 + 0.3 * coastal_factor) +
            enso_cycle[t] * (1 + 0.5 * coastal_factor) +
            ipo_cycle[t] +
            elevation_proxy * (-2) +
            np.random.normal(0, 0.3, (n_lats, n_lons))
        ).astype(np.float32)
        corr_noise = gaussian_filter(
            np.random.normal(0, 0.4, (n_lats, n_lons)), sigma=3
        ).astype(np.float32)
        tas[t] += corr_noise

    print(f"  Temperature: {tas.min():.1f} to {tas.max():.1f} °C")

    # --- Precipitation (pr) in mm/day ---
    pr_base = 6.0 + 3.0 * coastal_factor + 1.5 * np.cos(np.radians(lat_grid) * 2)

    pr = np.zeros((n_years, n_lats, n_lons), dtype=np.float32)
    for t in range(n_years):
        cc_scaling = 1.0 + 0.07 * temp_trend[t]
        monsoon_mod = 1.0 + 0.15 * enso_cycle[t]
        var_scale = 1.0 + 0.015 * t  # slower increase over 86 years

        pr[t] = (
            pr_base * monsoon_mod * cc_scaling +
            np.random.normal(0, 1.2 * var_scale, (n_lats, n_lons))
        ).astype(np.float32)
        corr_rain = gaussian_filter(
            np.random.normal(0, 1.5 * var_scale, (n_lats, n_lons)), sigma=5
        ).astype(np.float32)
        pr[t] += corr_rain

        # Extreme events
        if np.random.random() < 0.15:
            cx = np.random.randint(5, n_lats - 5)
            cy = np.random.randint(5, n_lons - 5)
            extreme = np.zeros((n_lats, n_lons), dtype=np.float32)
            extreme[cx-3:cx+3, cy-3:cy+3] = np.random.uniform(3, 8)
            extreme = gaussian_filter(extreme, sigma=2).astype(np.float32)
            pr[t] += extreme

        pr[t] = np.clip(pr[t], 0.01, 30)

    print(f"  Precipitation: {pr.min():.1f} to {pr.max():.1f} mm/day")

    # --- GDP ---
    gdp_base = np.random.uniform(3000, 15000, (n_lats, n_lons))
    cities = [
        (13.7, 100.5, 45000), (-6.2, 106.8, 35000), (14.6, 121.0, 28000),
        (10.8, 106.7, 30000), (1.3, 103.8, 60000), (3.1, 101.7, 40000),
        (7.0, 125.0, 20000), (-8.0, 112.0, 22000),
    ]
    for clat, clon, cgdp in cities:
        dist = np.sqrt((lat_grid - clat)**2 + (lon_grid - clon)**2)
        gdp_base += cgdp * np.exp(-dist**2 / 8)
    gdp = np.clip(gdp_base * (1 + 0.8 * coastal_factor), 1000, 80000).astype(np.float32)
    print(f"  GDP: ${gdp.min():.0f} to ${gdp.max():.0f} per capita")

    # --- Population ---
    pop_base = np.random.uniform(20, 500, (n_lats, n_lons))
    for clat, clon, cgdp in cities:
        dist = np.sqrt((lat_grid - clat)**2 + (lon_grid - clon)**2)
        pop_base += (cgdp / 10) * np.exp(-dist**2 / 5)
    pop = np.clip(pop_base * (1 + 1.5 * coastal_factor), 5, 15000).astype(np.float32)
    print(f"  Population: {pop.min():.0f} to {pop.max():.0f} per km²")

    # --- Soil moisture ---
    soil_moisture = np.clip(
        0.3 + 0.2 * (pr.mean(axis=0) / pr.mean()) + np.random.uniform(-0.05, 0.05, (n_lats, n_lons)),
        0.05, 0.95
    ).astype(np.float32)

    # --- Save ---
    data = {
        'tas': tas,
        'pr': pr,
        'gdp': gdp,
        'pop': pop,
        'soil_moisture': soil_moisture,
        'coastal_factor': coastal_factor.astype(np.float32),
        'lats': lats,
        'lons': lons,
        'years': years,
    }

    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'climate_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved to {pkl_path}")
    print(f"  Total nodes: {N}")
    print(f"  Total data points: {n_years * N * 2:,} (temp + precip)")
    print(f"  Years: {years[0]}-{years[-1]} ({n_years} years)")
    print(f"  Resolution: 0.5 degree (~55km at equator)")

    return data


if __name__ == '__main__':
    generate_synthetic_data()
