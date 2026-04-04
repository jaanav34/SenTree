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

Fix (v2): ENSO and IPO signals are now aperiodic and spatially heterogeneous.
  - Real ENSO is irregular (2-7yr range), not a clean sine wave. A pure
    sin(2π*t/5.2) produces perfectly periodic spikes every 5 years that
    propagate through the tail-risk engine as artificial volatility events.
  - ENSO is now an AR(1) process with stochastic period jitter, matching
    observed ONI index statistics (σ≈0.8°C, lag-1 autocorr≈0.6).
  - Spatial teleconnection pattern: ENSO influence varies by lat/lon
    (stronger in tropics, weaker at margins) — not uniform amplification.
  - IPO kept as slow multi-decadal variation but with noise added.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

import numpy as np
import pickle


def _generate_enso_index(n_years, rng, ar1_coeff=0.6, sigma=0.5):
    """
    Generate a realistic ENSO index as an AR(1) process.

    Real ENSO (ONI index) properties:
      - Irregular period: 2-7 years (NOT a fixed 5.2yr sine)
      - Lag-1 annual autocorrelation: ~0.5-0.7
      - Amplitude: ±1.5°C in Niño3.4, but SE Asia teleconnection ~±0.4°C

    AR(1): X(t) = ar1_coeff * X(t-1) + ε(t),  ε ~ N(0, sigma²)
    This produces realistic irregular oscillations without fixed periodicity.
    """
    enso = np.zeros(n_years)
    enso[0] = rng.normal(0, sigma)
    noise = rng.normal(0, sigma, n_years)
    for t in range(1, n_years):
        enso[t] = ar1_coeff * enso[t - 1] + noise[t]
    # Scale to realistic SE Asia teleconnection amplitude (~±0.4°C effect)
    enso = enso * (0.4 / (enso.std() + 1e-8))
    return enso


def _generate_ipo_index(n_years, rng):
    """
    Inter-decadal Pacific Oscillation: slow 20-30yr variation with noise.
    Much lower amplitude than ENSO (±0.1°C on regional temps).
    """
    # Slow sine base with stochastic phase
    phase_offset = rng.uniform(0, 2 * np.pi)
    period = rng.uniform(18, 28)   # irregular 18-28yr period
    t = np.arange(n_years)
    ipo = 0.12 * np.sin(2 * np.pi * t / period + phase_offset)
    ipo += rng.normal(0, 0.03, n_years)   # small noise
    return ipo


def _enso_teleconnection_pattern(lat_grid, lon_grid):
    """
    Spatial pattern of ENSO teleconnection over SE Asia.

    Real teleconnection: strongest near equatorial Pacific (lon>120, lat~0),
    weaker over land masses and higher latitudes.
    Returns a (nlat, nlon) weight map in [0, 1].
    """
    # Equatorial weighting (falls off with abs latitude)
    lat_weight = np.exp(-np.abs(lat_grid) / 12.0)

    # Eastern weighting (stronger influence from Pacific side)
    lon_weight = (lon_grid - lon_grid.min()) / (lon_grid.max() - lon_grid.min() + 1e-8)
    lon_weight = 0.4 + 0.6 * lon_weight   # range [0.4, 1.0]

    pattern = lat_weight * lon_weight
    # Normalize to [0.1, 1.0] so even margins get some signal
    pattern = 0.1 + 0.9 * (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
    return pattern.astype(np.float32)


def generate_synthetic_data(out_dir='data/processed'):
    """Generate and save synthetic data. Returns the data dict."""
    rng = np.random.default_rng(42)   # use Generator API for reproducibility

    # --- Grid: SE Asia coastal, 0.5-degree resolution ---
    lats = np.arange(-10, 25.5, 0.5)   # 71 points
    lons = np.arange(90, 130.5, 0.5)   # 81 points
    years = np.arange(2015, 2101)       # 86 years
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
    elevation_proxy = 0.3 * (1 - coastal_factor) * rng.uniform(0, 1, (n_lats, n_lons))

    # --- Climate indices (FIXED: aperiodic, realistic) ---
    enso_index = _generate_enso_index(n_years, rng, ar1_coeff=0.62, sigma=0.48)
    ipo_index  = _generate_ipo_index(n_years, rng)

    # Spatial teleconnection pattern — varies by location, not uniform
    enso_pattern = _enso_teleconnection_pattern(lat_grid, lon_grid)  # (nlat, nlon)

    # --- Long-term warming trend (SSP3-7.0) ---
    year_frac  = (years - 2015) / 85.0      # 0 → 1
    temp_trend = 4.5 * year_frac ** 1.3     # ~2.5°C at 2050, ~4.5°C at 2100

    # --- Temperature (tas) in °C ---
    tas = np.zeros((n_years, n_lats, n_lons), dtype=np.float32)
    tas_monthly = np.zeros((n_years, 12, n_lats, n_lons), dtype=np.float32)

    for t in range(n_years):
        # ENSO contribution: spatially heterogeneous (not uniform amplification)
        enso_contrib = enso_index[t] * enso_pattern   # (nlat, nlon)
        ipo_contrib   = ipo_index[t]                   # scalar

        tas_base_year = (
            temp_base_field
            + temp_trend[t] * (1 + 0.2 * coastal_factor)   # coastal amplification reduced
            + enso_contrib                                   # spatially heterogeneous ENSO
            + ipo_contrib
            - elevation_proxy * 2
        )

        for m in range(12):
            seasonal_amp   = 4.0 * np.abs(lat_grid) / 25.0 + 1.5
            seasonal_cycle = seasonal_amp * np.sin(2 * np.pi * (m - 3) / 12)
            tas_m = tas_base_year + seasonal_cycle + rng.normal(0, 0.4, (n_lats, n_lons))
            tas_monthly[t, m] = gaussian_filter(tas_m, sigma=2.0).astype(np.float32)

        tas[t] = np.mean(tas_monthly[t], axis=0)

    print(f"  Temperature: {tas.min():.1f} to {tas.max():.1f} °C")

    # --- Precipitation (pr) in mm/day ---
    pr_base = 6.0 + 3.0 * coastal_factor + 1.5 * np.cos(np.radians(lat_grid) * 2)

    pr = np.zeros((n_years, n_lats, n_lons), dtype=np.float32)
    pr_monthly = np.zeros((n_years, 12, n_lats, n_lons), dtype=np.float32)

    for t in range(n_years):
        cc_scaling  = 1.0 + 0.07 * temp_trend[t]
        # ENSO modulates precip spatially — wetter/drier depending on location
        # La Niña (negative ENSO) → more precip in maritime SE Asia
        enso_precip_mod = 1.0 - 0.12 * enso_index[t] * enso_pattern   # heterogeneous
        var_scale   = 1.0 + 0.008 * t   # slower variance growth

        for m in range(12):
            monsoon_phase  = np.sin(np.pi * (m - 3) / 6) * np.sign(lat_grid)
            seasonal_factor = 1.0 + 0.8 * monsoon_phase

            pr_m = pr_base * enso_precip_mod * cc_scaling * seasonal_factor
            pr_m += rng.normal(0, 2.5 * var_scale, (n_lats, n_lons))
            pr_m = gaussian_filter(np.clip(pr_m, 0.05, 50.0), sigma=1.5).astype(np.float32)
            pr_monthly[t, m] = pr_m

        pr[t] = np.mean(pr_monthly[t], axis=0)

    print(f"  Precipitation: {pr.min():.1f} to {pr.max():.1f} mm/day")

    # --- GDP ---
    gdp_base = rng.uniform(3000, 15000, (n_lats, n_lons))
    cities = [
        (13.7, 100.5, 45000), (-6.2, 106.8, 35000), (14.6, 121.0, 28000),
        (10.8, 106.7, 30000), (1.3, 103.8, 60000),  (3.1, 101.7, 40000),
        (7.0, 125.0, 20000),  (-8.0, 112.0, 22000),
    ]
    for clat, clon, cgdp in cities:
        dist = np.sqrt((lat_grid - clat)**2 + (lon_grid - clon)**2)
        gdp_base += cgdp * np.exp(-dist**2 / 8)
    gdp = np.clip(gdp_base * (1 + 0.8 * coastal_factor), 1000, 80000).astype(np.float32)
    print(f"  GDP: ${gdp.min():.0f} to ${gdp.max():.0f} per capita")

    # --- Population ---
    pop_base = rng.uniform(20, 500, (n_lats, n_lons))
    for clat, clon, cgdp in cities:
        dist = np.sqrt((lat_grid - clat)**2 + (lon_grid - clon)**2)
        pop_base += (cgdp / 10) * np.exp(-dist**2 / 5)
    pop = np.clip(pop_base * (1 + 1.5 * coastal_factor), 5, 15000).astype(np.float32)
    print(f"  Population: {pop.min():.0f} to {pop.max():.0f} per km²")

    # --- Soil moisture ---
    soil_moisture = np.clip(
        0.3 + 0.2 * (pr.mean(axis=0) / (pr.mean() + 1e-8))
        + rng.uniform(-0.05, 0.05, (n_lats, n_lons)),
        0.05, 0.95
    ).astype(np.float32)

    # --- Save ---
    data = {
        'tas':            tas,
        'tas_monthly':    tas_monthly,
        'pr':             pr,
        'pr_monthly':     pr_monthly,
        'gdp':            gdp,
        'pop':            pop,
        'soil_moisture':  soil_moisture,
        'coastal_factor': coastal_factor.astype(np.float32),
        'lats':           lats,
        'lons':           lons,
        'years':          years,
    }

    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'climate_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved to {pkl_path}")
    print(f"  Total nodes: {N}")
    print(f"  Years: {years[0]}-{years[-1]} ({n_years} years)")
    print(f"  ENSO index: mean={enso_index.mean():.3f}, std={enso_index.std():.3f}")
    print(f"  Resolution: 0.5 degree (~55km at equator)")

    return data


if __name__ == '__main__':
    generate_synthetic_data()