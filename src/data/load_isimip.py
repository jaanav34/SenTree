"""
Load ISIMIP3b NetCDF data or fall back to synthetic.

Handles the real ISIMIP3b dataset structure:
  - Variable: tas (Near-Surface Air Temperature, units: Kelvin)
  - Dims: (time, lat, lon) — DAILY data
  - Resolution: 0.5° globally (360 lat x 720 lon)
  - Lat: 89.75 to -89.75 (DESCENDING)
  - Lon: -179.75 to 179.75
  - Files split by decade: *_2015_2020.nc, *_2021_2030.nc, ...

We resample daily -> annual mean, subset to SE Asia, convert K -> °C.
"""
import os
import glob
import pickle
import numpy as np


# SE Asia coastal bounding box
LAT_MIN, LAT_MAX = -10.0, 25.0
LON_MIN, LON_MAX = 90.0, 130.0


def load_climate_data(data_dir='data/processed', raw_dir='data/raw'):
    """Load processed climate data. Try real ISIMIP, fall back to synthetic."""
    pkl_path = os.path.join(data_dir, 'climate_data.pkl')

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cached data from {pkl_path}")
        return data

    # Try to find ISIMIP NetCDF files
    nc_files = _find_isimip_files(raw_dir)
    if nc_files:
        print(f"  Found {len(nc_files)} ISIMIP NetCDF files in {raw_dir}")
        return _load_from_netcdf(nc_files, data_dir)

    # Also check parent directory (cluster layout: ../data/raw)
    alt_raw = os.path.join(os.path.dirname(raw_dir), '..', 'data', 'raw')
    nc_files = _find_isimip_files(alt_raw)
    if nc_files:
        print(f"  Found {len(nc_files)} ISIMIP NetCDF files in {alt_raw}")
        return _load_from_netcdf(nc_files, data_dir)

    # Fall back to synthetic
    print("  WARNING: No ISIMIP data found. Generating synthetic data.")
    return _generate_synthetic(data_dir)


def _find_isimip_files(raw_dir):
    """Find all ISIMIP tas NetCDF files in a directory."""
    if not os.path.isdir(raw_dir):
        return []
    patterns = [
        os.path.join(raw_dir, '*ssp370_tas_global_daily_*.nc'),
        os.path.join(raw_dir, '*ssp370_tas_*.nc'),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))


def _load_from_netcdf(nc_files, out_dir):
    """
    Load multiple ISIMIP3b daily NetCDF files, concatenate, resample to
    annual means, subset to SE Asia, convert K -> °C.
    """
    import xarray as xr

    print(f"  Opening {len(nc_files)} files...")
    for f in nc_files:
        print(f"    {os.path.basename(f)}")

    # Open all files as a single dataset (auto-concatenates along time)
    ds = xr.open_mfdataset(nc_files, combine='by_coords', chunks={'time': 365})

    # The variable is 'tas' (Near-Surface Air Temperature in Kelvin)
    tas_raw = ds['tas']
    print(f"  Raw shape: {dict(tas_raw.sizes)}")
    print(f"  Time range: {str(tas_raw.time.values[0])[:10]} to {str(tas_raw.time.values[-1])[:10]}")

    # Subset to SE Asia
    # ISIMIP lat is DESCENDING (89.75, 89.25, ..., -89.75), so we need
    # to handle the slice direction correctly
    lat_vals = tas_raw.lat.values
    if lat_vals[0] > lat_vals[-1]:
        # Descending lat: slice(max, min)
        tas_subset = tas_raw.sel(
            lat=slice(LAT_MAX, LAT_MIN),
            lon=slice(LON_MIN, LON_MAX),
        )
    else:
        tas_subset = tas_raw.sel(
            lat=slice(LAT_MIN, LAT_MAX),
            lon=slice(LON_MIN, LON_MAX),
        )

    print(f"  SE Asia subset: {dict(tas_subset.sizes)}")

    # Resample daily -> monthly mean
    print("  Resampling daily -> monthly mean...")
    tas_monthly = tas_subset.resample(time='1MS').mean()
    
    # Resample daily -> annual mean
    print("  Resampling daily -> annual mean...")
    tas_annual = tas_subset.resample(time='1YE').mean()

    # Load into memory and convert K -> °C
    print("  Loading into memory & converting K -> °C...")
    tas_annual_values = tas_annual.values - 273.15
    tas_monthly_values = tas_monthly.values - 273.15

    # Get coordinates
    lats = tas_annual.lat.values.astype(np.float64)
    lons = tas_annual.lon.values.astype(np.float64)

    # Ensure lat is ascending (for consistent plotting with origin='lower')
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        tas_annual_values = tas_annual_values[:, ::-1, :]
        tas_monthly_values = tas_monthly_values[:, :, ::-1, :]

    # Extract years from the time coordinate
    years = np.array([int(str(t)[:4]) for t in tas_annual.time.values])
    T_years = len(years)

    # Reshape monthly values to (T_years, 12, nlat, nlon)
    # Note: This assumes complete years in the dataset
    nlat, nlon = len(lats), len(lons)
    tas_monthly_values = tas_monthly_values[:T_years*12].reshape(T_years, 12, nlat, nlon)

    print(f"  Annual data: {tas_annual_values.shape} — {len(years)} years")
    print(f"  Monthly data: {tas_monthly_values.shape}")

    # We don't have pr (precipitation) in the downloaded files — synthesize it
    print("  Generating synthetic precipitation (monthly + annual)...")
    pr_monthly = _synthesize_monthly_precipitation(tas_monthly_values, lats, lons, years)
    pr_annual = np.mean(pr_monthly, axis=1) # (T, nlat, nlon)

    # GDP and population (gridded proxies for SE Asia)
    gdp, pop, soil_moisture, coastal_factor = _generate_socioeconomic(lats, lons)

    data = {
        'tas': tas_annual_values.astype(np.float32),
        'tas_monthly': tas_monthly_values.astype(np.float32),
        'pr': pr_annual.astype(np.float32),
        'pr_monthly': pr_monthly.astype(np.float32),
        'gdp': gdp.astype(np.float32),
        'pop': pop.astype(np.float32),
        'soil_moisture': soil_moisture.astype(np.float32),
        'coastal_factor': coastal_factor.astype(np.float32),
        'lats': lats,
        'lons': lons,
        'years': years,
    }

    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'climate_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved processed data to {pkl_path}")

    ds.close()
    return data


def _synthesize_monthly_precipitation(tas_monthly, lats, lons, years):
    """
    Generate realistic monthly precipitation from monthly temperature
    incorporating SE Asian monsoon seasonality.
    """
    from scipy.ndimage import gaussian_filter

    np.random.seed(42)
    T, M, nlat, nlon = tas_monthly.shape
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    # SE Asian Monsoon: peak in Jun-Sep (NH) or Nov-Feb (SH)
    # We'll use a latitudinal shift for seasonality
    month_idx = np.arange(12)
    
    # Coastal and tropical base
    coastal_distance = (lon_grid.max() - lon_grid) / (lon_grid.max() - lon_grid.min() + 1e-8)
    coastal_factor = np.exp(-3 * coastal_distance)
    
    pr_monthly = np.zeros_like(tas_monthly)
    
    for t in range(T):
        for m in range(12):
            # Monsoon phase depends on latitude (NH peak in Jul, SH peak in Jan)
            monsoon_phase = np.sin(np.pi * (m - 3) / 6) * np.sign(lat_grid)
            seasonal_factor = 1.0 + 0.8 * monsoon_phase
            
            # Clausius-Clapeyron scaling relative to annual start
            temp_anomaly = tas_monthly[t, m] - tas_monthly[0].mean(axis=0)
            cc_scaling = 1.0 + 0.07 * np.clip(temp_anomaly, -10, 15)
            
            pr_m = (5.0 + 4.0 * coastal_factor) * seasonal_factor * cc_scaling
            noise = np.random.normal(0, 2.0, (nlat, nlon))
            pr_m = np.clip(pr_m + noise, 0.05, 50.0)
            
            # Spatial smoothing for realistic rain bands
            pr_monthly[t, m] = gaussian_filter(pr_m, sigma=1.5)

    return pr_monthly


def _generate_socioeconomic(lats, lons):
    """Generate GDP, population, soil moisture, and coastal factor grids."""
    from scipy.ndimage import gaussian_filter

    np.random.seed(123)
    nlat, nlon = len(lats), len(lons)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    # Coastal factor
    coastal_distance = (lon_grid.max() - lon_grid) / (lon_grid.max() - lon_grid.min())
    coastal_factor = np.exp(-3 * coastal_distance)

    # GDP
    gdp_base = np.random.uniform(3000, 15000, (nlat, nlon))
    cities = [
        (13.7, 100.5, 45000), (-6.2, 106.8, 35000), (14.6, 121.0, 28000),
        (10.8, 106.7, 30000), (1.3, 103.8, 60000), (3.1, 101.7, 40000),
        (7.0, 125.0, 20000), (-8.0, 112.0, 22000),
    ]
    for clat, clon, cgdp in cities:
        dist = np.sqrt((lat_grid - clat)**2 + (lon_grid - clon)**2)
        gdp_base += cgdp * np.exp(-dist**2 / 8)
    gdp = np.clip(gdp_base * (1 + 0.8 * coastal_factor), 1000, 80000)

    # Population
    pop_base = np.random.uniform(20, 500, (nlat, nlon))
    for clat, clon, cgdp in cities:
        dist = np.sqrt((lat_grid - clat)**2 + (lon_grid - clon)**2)
        pop_base += (cgdp / 10) * np.exp(-dist**2 / 5)
    pop = np.clip(pop_base * (1 + 1.5 * coastal_factor), 5, 15000)

    # Soil moisture
    soil = np.clip(0.3 + 0.15 * coastal_factor + np.random.uniform(-0.05, 0.05, (nlat, nlon)), 0.05, 0.95)

    return gdp, pop, soil, coastal_factor


def _generate_synthetic(out_dir):
    """Generate synthetic data matching real ISIMIP structure. 2015-2100."""
    # Import the standalone generator
    from data.generate_synthetic import generate_synthetic_data
    return generate_synthetic_data(out_dir)
