"""
Generate high-fidelity synthetic ISIMIP-like climate data for SE Asia.

v3 improvements over v2:
  1.  Land/sea mask — ocean cells get different temperature and
      precipitation physics than land cells.
  2.  Orography (elevation) — realistic highland cooling using a
      parametric mountain field for SE Asian highlands (Mekong
      highlands, Borneo interior, Philippine Cordillera, etc.).
      Lapse rate: -6.5°C / 1000m (ICAO standard).
  3.  SST-driven coastal temperatures — coastal land cells warm more
      slowly than interior cells (ocean thermal inertia).
  4.  Multi-scale precipitation:
        - ITCZ migrates northward ~1° by 2100 (SSP3-7.0).
        - Orographic enhancement on windward slopes.
        - Asymmetric monsoon withdrawal by hemisphere.
        - Clausius-Clapeyron scaling per-cell based on local warming.
  5.  AR(2) ENSO + IPO + PDO — PDO modulates ENSO amplitude.
  6.  Spatially correlated noise via FFT Gaussian random fields.
  7.  Soil moisture from a simple bucket model (ET + drainage).
  8.  More cities, land-masked population (no ocean population).
  9.  Elevation and land_mask exported to data dict.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv
ensure_venv()

import numpy as np
import pickle
from scipy.ndimage import gaussian_filter


LAPSE_RATE = 6.5 / 1000.0   # °C per metre


# ============================================================
# Climate index generators
# ============================================================

def _ar2_process(n, rng, phi1=0.9, phi2=-0.3, sigma=0.4):
    """AR(2) — richer spectral structure than AR(1)."""
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma)
    x[1] = phi1 * x[0] + rng.normal(0, sigma)
    noise = rng.normal(0, sigma, n)
    for t in range(2, n):
        x[t] = phi1 * x[t-1] + phi2 * x[t-2] + noise[t]
    return x


def _generate_enso_index(n_years, rng):
    """
    ENSO as AR(2) capturing 2-7yr irregular period of observed ONI.
    Scaled to ±0.45°C SE Asia teleconnection amplitude.
    """
    raw = _ar2_process(n_years, rng, phi1=0.88, phi2=-0.32, sigma=0.42)
    return raw * (0.45 / (raw.std() + 1e-8))


def _generate_ipo_index(n_years, rng):
    """IPO: slow AR(2) + low-frequency sine, amplitude ±0.12°C."""
    ar = _ar2_process(n_years, rng, phi1=0.95, phi2=0.0, sigma=0.05)
    t  = np.arange(n_years)
    slow = 0.10 * np.sin(2 * np.pi * t / rng.uniform(20, 28) + rng.uniform(0, 2*np.pi))
    combined = ar + slow
    return combined * (0.12 / (combined.std() + 1e-8))


def _generate_pdo_index(n_years, rng):
    """PDO: 15-25yr sine + noise, amplitude ±0.08°C."""
    t   = np.arange(n_years)
    pdo = 0.08 * np.sin(2 * np.pi * t / rng.uniform(15, 25) + rng.uniform(0, 2*np.pi))
    pdo += rng.normal(0, 0.02, n_years)
    return pdo


# ============================================================
# Spatial field builders
# ============================================================

def _build_orography(lat_grid, lon_grid, rng):
    """Parametric elevation field (m) for SE Asia mountain ranges."""
    elev = np.zeros_like(lat_grid, dtype=np.float64)
    mountains = [
        # (clat, clon, peak_m, lat_sigma, lon_sigma)
        (20.0, 103.0, 2500, 3.0, 3.0),   # Yunnan-Mekong highlands
        (16.5, 107.5, 1800, 2.0, 1.5),   # Annamite Range
        (15.0, 101.5, 1200, 2.5, 1.5),   # Thai highlands
        ( 4.5, 116.5, 2000, 2.0, 2.0),   # Borneo / Mt Kinabalu
        ( 1.5, 110.5,  900, 1.5, 1.5),   # Sarawak highlands
        (16.5, 121.0, 1500, 1.5, 1.0),   # Philippine Cordillera
        (10.0, 123.5,  800, 1.5, 1.0),   # Visayas highlands
        ( 8.5, 125.5, 1200, 1.5, 1.0),   # Mindanao highlands
        (13.0,  99.5,  700, 1.5, 1.0),   # Tenasserim Hills
        (23.0, 120.5, 2500, 2.0, 1.5),   # Taiwan Central Range
    ]
    for clat, clon, peak, slat, slon in mountains:
        d2 = ((lat_grid - clat) / slat)**2 + ((lon_grid - clon) / slon)**2
        elev += peak * np.exp(-d2)
    rough = gaussian_filter(rng.uniform(0, 200, lat_grid.shape), sigma=3)
    elev  = np.clip(elev + rough * (1 - np.exp(-elev / 500)), 0, 4000)
    return elev.astype(np.float32)


def _build_land_mask(lat_grid, lon_grid):
    """Approximate land mask for SE Asia (1=land, 0=ocean)."""
    m = np.zeros_like(lat_grid, dtype=np.float32)
    regions = [
        dict(lat=(1, 24),  lon=(97, 110)),   # Mainland SE Asia
        dict(lat=(1,  7),  lon=(99, 104)),   # Malay Peninsula
        dict(lat=(-6, 6),  lon=(95, 107)),   # Sumatra
        dict(lat=(-9,-6),  lon=(105,115)),   # Java
        dict(lat=(-4, 7),  lon=(108,119)),   # Borneo
        dict(lat=(-5, 2),  lon=(119,125)),   # Sulawesi
        dict(lat=( 5,19),  lon=(118,127)),   # Philippines
        dict(lat=( 8,23),  lon=(103,110)),   # Vietnam coast
        dict(lat=(10,24),  lon=( 92,101)),   # Myanmar
        dict(lat=(-9,-5),  lon=(115,116)),   # Bali/Lombok
        dict(lat=(23,25),  lon=(120,122)),   # Taiwan
    ]
    for r in regions:
        m[(lat_grid > r['lat'][0]) & (lat_grid < r['lat'][1]) &
          (lon_grid > r['lon'][0]) & (lon_grid < r['lon'][1])] = 1.0
    return np.clip(gaussian_filter(m, sigma=1.5), 0, 1)


def _build_coastal_factor(land_mask):
    """Exponential decay from land-sea boundary."""
    from scipy.ndimage import distance_transform_edt
    ocean = (land_mask < 0.5).astype(np.float32)
    dist  = distance_transform_edt(1 - ocean)
    return np.exp(-dist / 8.0).astype(np.float32)


def _gaussian_random_field(shape, rng, corr_length=5.0):
    """Spatially correlated noise — avoids white-noise texture."""
    return gaussian_filter(rng.standard_normal(shape), sigma=corr_length)


# ============================================================
# Soil moisture bucket model
# ============================================================

def _soil_moisture_bucket(pr_monthly, tas_monthly, lats,
                           field_capacity=0.6, wilting_point=0.05,
                           drainage_rate=0.15):
    """
    Single-bucket model: S(t) = S(t-1) + P - ET - drainage.
    PET via temperature-based Hamon method.
    Returns (T, nlat, nlon) normalised to [0,1].
    """
    T, M, nlat, nlon = pr_monthly.shape
    lat_arr = np.array(lats)[:, None] * np.ones((nlat, nlon))
    day_len = 1.0 + 0.5 * np.cos(np.radians(lat_arr))

    soil = np.full((nlat, nlon), 0.4)
    out  = np.zeros((T, nlat, nlon), dtype=np.float32)

    for t in range(T):
        monthly = []
        for m in range(M):
            p_in  = pr_monthly[t, m] / 30.0 * 0.08
            tc    = tas_monthly[t, m]
            e_sat = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
            pet   = 0.0065 * day_len * e_sat / 30.0
            et    = np.minimum(pet, soil * 0.7)
            drain = drainage_rate * np.maximum(soil - field_capacity, 0)
            soil  = np.clip(soil + p_in - et - drain,
                            wilting_point, field_capacity)
            monthly.append(soil.copy())
        out[t] = np.mean(monthly, axis=0)

    return np.clip(
        (out - wilting_point) / (field_capacity - wilting_point + 1e-8),
        0, 1
    ).astype(np.float32)


# ============================================================
# Main generator
# ============================================================

def generate_synthetic_data(out_dir='data/processed'):
    """Generate and save synthetic SE Asia climate data. Returns data dict."""
    rng = np.random.default_rng(42)

    lats  = np.arange(-10, 25.5, 0.5)
    lons  = np.arange( 90, 130.5, 0.5)
    years = np.arange(2015, 2101)
    n_lats, n_lons, n_years = len(lats), len(lons), len(years)
    N = n_lats * n_lons

    print(f"Generating synthetic data v3: {n_years} years "
          f"({years[0]}-{years[-1]}), {n_lats}×{n_lons} = {N} nodes")

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    # ---- Spatial infrastructure ----
    print("  Building spatial fields (land mask, orography, coastal)...")
    land_mask      = _build_land_mask(lat_grid, lon_grid)
    coastal_factor = _build_coastal_factor(land_mask)
    elevation      = _build_orography(lat_grid, lon_grid, rng)
    elev_cooling   = elevation * LAPSE_RATE

    # Ocean thermal inertia: ocean/coastal cells warm more slowly
    ocean_buffer = np.clip(
        1.0 - 0.35 * (1 - land_mask) - 0.15 * coastal_factor,
        0.5, 1.0
    )
    # Spatially varying warming amplification (land > ocean)
    warming_pattern = gaussian_filter(
        1.0 + 0.3 * land_mask - 0.15 * (1 - land_mask),
        sigma=2
    )

    # ---- Climate indices ----
    print("  Generating climate indices (AR2 ENSO + IPO + PDO)...")
    enso = _generate_enso_index(n_years, rng)
    ipo  = _generate_ipo_index(n_years, rng)
    pdo  = _generate_pdo_index(n_years, rng)
    enso_amp = enso * (1.0 + 0.3 * pdo)   # PDO modulates ENSO amplitude

    lat_w  = np.exp(-np.abs(lat_grid) / 12.0)
    lon_w  = 0.4 + 0.6 * (lon_grid - lon_grid.min()) / (lon_grid.max() - lon_grid.min() + 1e-8)
    raw_p  = lat_w * lon_w
    enso_pattern = (0.1 + 0.9 * (raw_p - raw_p.min()) /
                    (raw_p.max() - raw_p.min() + 1e-8)).astype(np.float32)

    # ---- Warming trend ----
    year_frac  = (years - 2015) / 85.0
    temp_trend = 4.5 * year_frac ** 1.3    # SSP3-7.0

    # ---- Temperature base field ----
    temp_base = (
        30.0
        - 0.45 * np.abs(lat_grid)
        - elev_cooling
        + 1.5 * (1 - land_mask)            # SST warmer than land surface
        + gaussian_filter(
            rng.uniform(-2, 2, (n_lats, n_lons)), sigma=4
          )
    )

    # ---- Temperature time series ----
    print("  Generating temperature...")
    tas         = np.zeros((n_years, n_lats, n_lons), dtype=np.float32)
    tas_monthly = np.zeros((n_years, 12, n_lats, n_lons), dtype=np.float32)

    for t in range(n_years):
        spatial_noise = _gaussian_random_field((n_lats, n_lons), rng, 6.0) * 0.3
        tas_year = (
            temp_base
            + temp_trend[t] * warming_pattern * ocean_buffer
            + enso_amp[t] * enso_pattern
            + ipo[t]
            + pdo[t] * 0.4
            + spatial_noise
        )
        for m in range(12):
            seasonal_amp = 3.5 * np.abs(lat_grid) / 25.0 + 1.0 + 1.5 * land_mask
            seasonal     = seasonal_amp * np.sin(2 * np.pi * (m - 3) / 12)
            diurnal      = rng.normal(0, 0.3 + 0.3 * land_mask, (n_lats, n_lons))
            tas_monthly[t, m] = gaussian_filter(
                (tas_year + seasonal + diurnal).astype(np.float32), sigma=1.5
            )
        tas[t] = tas_monthly[t].mean(axis=0)

    print(f"  Temperature: {tas.min():.1f} to {tas.max():.1f} °C")

    # ---- Precipitation base ----
    itcz_base  = 8.0 * np.exp(-((lat_grid - 2.0) / 6.0)**2)
    orog_pr    = np.clip(elevation / 1000.0, 0, 3) * 2.0
    pr_base    = np.clip(
        itcz_base + 3.5 * coastal_factor + orog_pr
        + 1.5 * np.cos(np.radians(lat_grid) * 2),
        0.5, 25.0
    )
    itcz_shift = 1.0 * year_frac   # ITCZ migrates ~1° north by 2100

    # ---- Precipitation time series ----
    print("  Generating precipitation...")
    pr         = np.zeros((n_years, n_lats, n_lons), dtype=np.float32)
    pr_monthly = np.zeros((n_years, 12, n_lats, n_lons), dtype=np.float32)

    for t in range(n_years):
        local_warming = temp_trend[t] * warming_pattern * ocean_buffer
        cc_scale      = 1.0 + 0.07 * local_warming
        enso_pr       = 1.0 - 0.15 * enso_amp[t] * enso_pattern
        itcz_mod      = np.clip(
            np.exp(-((lat_grid - 2.0 - itcz_shift[t]) / 6.0)**2) /
            (np.exp(-((lat_grid - 2.0) / 6.0)**2) + 1e-8),
            0.8, 1.2
        )
        var_scale = 1.0 + 0.012 * t

        for m in range(12):
            nh = (lat_grid > 0).astype(float)
            sh = 1 - nh
            nh_mon = np.sin(np.pi * np.clip(m - 4, 0, 4) / 4)
            sh_mon = np.sin(np.pi * np.clip((m + 6) % 12 - 4, 0, 4) / 4)
            monsoon = 1.0 + 0.9 * nh * nh_mon + 0.9 * sh * sh_mon
            dry_sup = 1.0 - 0.4 * nh * (1 - np.clip(np.sin(np.pi * m / 5), 0, 1))

            noise = _gaussian_random_field((n_lats, n_lons), rng, 4.0) * 2.0 * var_scale
            pr_m  = pr_base * cc_scale * enso_pr * monsoon * dry_sup * itcz_mod
            pr_monthly[t, m] = gaussian_filter(
                np.clip(pr_m + noise, 0.05, 60.0).astype(np.float32),
                sigma=1.2
            )

        pr[t] = pr_monthly[t].mean(axis=0)

    print(f"  Precipitation: {pr.min():.1f} to {pr.max():.1f} mm/day")

    # ---- Soil moisture ----
    print("  Computing soil moisture (bucket model)...")
    soil_series   = _soil_moisture_bucket(pr_monthly, tas_monthly, lats)
    soil_moisture = soil_series[-1]   # static snapshot for feature matrix

    # ---- GDP ----
    print("  Building GDP + population...")
    gdp_base = rng.uniform(3000, 15000, (n_lats, n_lons))
    cities = [
        (13.7, 100.5, 45000), (-6.2, 106.8, 35000), (14.6, 121.0, 28000),
        (10.8, 106.7, 30000), ( 1.3, 103.8, 65000), ( 3.1, 101.7, 42000),
        ( 7.0, 125.0, 20000), (-8.0, 112.0, 22000), (16.8,  96.2, 18000),
        (11.6, 104.9, 15000), (17.9, 102.6, 14000), (15.0,  99.9, 16000),
        ( 1.5, 110.3, 20000), ( 5.4, 100.3, 28000),
    ]
    for clat, clon, cgdp in cities:
        dist = np.sqrt((lat_grid - clat)**2 + (lon_grid - clon)**2)
        gdp_base += cgdp * np.exp(-dist**2 / 6)
    gdp = np.clip(
        gdp_base * (1 + 0.6 * coastal_factor + 0.3 * land_mask),
        1000, 85000
    ).astype(np.float32)
    print(f"  GDP: ${gdp.min():.0f} to ${gdp.max():.0f} per capita")

    # ---- Population (land-masked — no ocean population) ----
    pop_base = rng.uniform(10, 300, (n_lats, n_lons))
    for clat, clon, cgdp in cities:
        dist = np.sqrt((lat_grid - clat)**2 + (lon_grid - clon)**2)
        pop_base += (cgdp / 8) * np.exp(-dist**2 / 4)
    pop = np.clip(
        pop_base * (1 + 1.2 * coastal_factor) * land_mask,
        0, 18000
    ).astype(np.float32)
    print(f"  Population: {pop.min():.0f} to {pop.max():.0f} per km²")

    # ---- Summary ----
    print(f"\n  ENSO: std={enso.std():.3f}, range=[{enso.min():.2f},{enso.max():.2f}]")
    print(f"  IPO:  std={ipo.std():.3f}  |  PDO: std={pdo.std():.3f}")
    print(f"  Elevation: {elevation.min():.0f}m to {elevation.max():.0f}m")
    print(f"  Land fraction: {land_mask.mean():.2f}")

    # ---- Save ----
    data = {
        'tas':                  tas,
        'tas_monthly':          tas_monthly,
        'pr':                   pr,
        'pr_monthly':           pr_monthly,
        'gdp':                  gdp,
        'pop':                  pop,
        'soil_moisture':        soil_moisture,
        'soil_moisture_series': soil_series,
        'coastal_factor':       coastal_factor,
        'land_mask':            land_mask,
        'elevation':            elevation,
        'lats':                 lats,
        'lons':                 lons,
        'years':                years,
    }

    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'climate_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved to {pkl_path}")
    print(f"  Nodes: {N}  |  Years: {years[0]}-{years[-1]}")
    print(f"  Resolution: 0.5° (~55km at equator)")
    return data


if __name__ == '__main__':
    generate_synthetic_data()