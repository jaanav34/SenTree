"""
Load ISIMIP3b NetCDF data or fall back to synthetic.

Handles the real ISIMIP3b dataset structure:
  - Variable: tas (Near-Surface Air Temperature, units: Kelvin)
  - Variable: pr (Precipitation, often kg m-2 s-1)
  - Dims: (time, lat, lon) — DAILY data
  - Resolution: 0.5° globally (360 lat x 720 lon)
  - Lat: 89.75 to -89.75 (DESCENDING)
  - Lon: -179.75 to 179.75
  - Files split by decade: *_2015_2020.nc, *_2021_2030.nc, ...

We resample daily -> annual mean, subset to SE Asia, convert:
  - tas: K -> °C
  - pr:  kg m-2 s-1 -> mm/day (×86400), when applicable
"""
import os
import glob
import pickle
import numpy as np


# Default SE Asia coastal bounding box
LAT_MIN, LAT_MAX = -10.0, 25.0
LON_MIN, LON_MAX = 90.0, 130.0


def load_climate_data(
    data_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    *,
    region: str = "se_asia",
    coarsen: int = 1,
):
    """
    Load processed climate data. Try real ISIMIP, fall back to synthetic.

    Args:
        data_dir: output directory for cached pickle
        raw_dir: directory containing ISIMIP NetCDFs
        region: "se_asia" (default) or "global"
        coarsen: integer factor to coarsen lat/lon (>=1). Use 2/4 for faster global runs.
    """
    if region not in {"se_asia", "global"}:
        raise ValueError(f"Unknown region: {region} (expected 'se_asia' or 'global')")
    if coarsen < 1:
        raise ValueError("coarsen must be >= 1")

    cache_name = f"climate_data_{region}_c{int(coarsen)}.pkl"
    pkl_path = os.path.join(data_dir, cache_name)

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cached data from {pkl_path}")
        return data

    # Try to find ISIMIP NetCDF files (tas required; pr optional)
    tas_files = _find_isimip_files(raw_dir, var="tas")
    pr_files = _find_isimip_files(raw_dir, var="pr")
    if tas_files:
        print(f"  Found {len(tas_files)} ISIMIP tas NetCDF files in {raw_dir}")
        if pr_files:
            print(f"  Found {len(pr_files)} ISIMIP pr NetCDF files in {raw_dir}")
        return _load_from_netcdf(tas_files, pr_files, data_dir, region=region, coarsen=coarsen)

    # Also check parent directory (cluster layout: ../data/raw)
    alt_raw = os.path.join(os.path.dirname(raw_dir), '..', 'data', 'raw')
    tas_files = _find_isimip_files(alt_raw, var="tas")
    pr_files = _find_isimip_files(alt_raw, var="pr")
    if tas_files:
        print(f"  Found {len(tas_files)} ISIMIP tas NetCDF files in {alt_raw}")
        if pr_files:
            print(f"  Found {len(pr_files)} ISIMIP pr NetCDF files in {alt_raw}")
        return _load_from_netcdf(tas_files, pr_files, data_dir, region=region, coarsen=coarsen)

    # Fall back to synthetic
    print("  WARNING: No ISIMIP data found. Generating synthetic data.")
    return _generate_synthetic(data_dir)


def _find_isimip_files(raw_dir, *, var: str):
    """Find ISIMIP NetCDF files for a variable in a directory."""
    if not os.path.isdir(raw_dir):
        return []
    if var not in {"tas", "pr"}:
        raise ValueError(f"Unsupported ISIMIP variable: {var}")
    patterns = [
        os.path.join(raw_dir, f'*ssp370_{var}_global_daily_*.nc'),
        os.path.join(raw_dir, f'*ssp370_{var}_*.nc'),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))


def _load_from_netcdf(tas_files, pr_files, out_dir, *, region: str, coarsen: int):
    """
    Load ISIMIP3b daily NetCDF files for tas (and optionally pr), concatenate,
    resample to annual means, and optionally subset/coarsen.
    """
    import xarray as xr

    def _open_var(files, varname: str):
        print(f"  Opening {varname} ({len(files)} files)...")
        for f in files:
            print(f"    {os.path.basename(f)}")

        # `xarray.open_mfdataset(..., chunks=...)` requires dask. This repo does not
        # require dask, so we fall back gracefully when it isn't available.
        chunks = None
        try:
            import importlib.util

            if importlib.util.find_spec("dask") is not None:
                chunks = {"time": 365}
        except Exception:
            chunks = None

        try:
            ds = xr.open_mfdataset(files, combine='by_coords', chunks=chunks)
        except ValueError as e:
            # Common message: "unrecognized chunk manager dask"
            msg = str(e).lower()
            if "dask" in msg or "chunk" in msg:
                ds = xr.open_mfdataset(files, combine='by_coords')
            else:
                raise
        if varname not in ds:
            raise KeyError(f"Variable '{varname}' not found in dataset. Variables: {list(ds.data_vars)}")
        da = ds[varname]
        units = da.attrs.get("units", "")
        if units:
            print(f"  {varname} units: {units}")
        return ds, da

    ds_tas, tas_raw = _open_var(tas_files, "tas")
    print(f"  Raw shape: {dict(tas_raw.sizes)}")
    print(f"  Time range: {str(tas_raw.time.values[0])[:10]} to {str(tas_raw.time.values[-1])[:10]}")

    # ISIMIP lat is often DESCENDING (89.75, 89.25, ..., -89.75), so we need
    # to handle slice direction correctly when slicing.
    lat_vals = tas_raw.lat.values

    if region == "se_asia":
        if lat_vals[0] > lat_vals[-1]:
            # Descending lat: slice(max, min)
            tas_subset = tas_raw.sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))
        else:
            tas_subset = tas_raw.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
        print(f"  SE Asia subset (tas): {dict(tas_subset.sizes)}")
    else:
        tas_subset = tas_raw
        print(f"  Global (tas): {dict(tas_subset.sizes)}")

    if coarsen > 1:
        print(f"  Coarsening tas by factor {coarsen} (lat/lon)...")
        tas_subset = tas_subset.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

    # Resample daily -> annual mean
    print("  Resampling daily -> annual mean (tas)...")
    tas_annual = tas_subset.resample(time='1YE').mean()

    # Load into memory and convert K -> °C
    print("  Loading into memory & converting K -> °C...")
    tas_values = tas_annual.values - 273.15  # K -> °C

    # Get coordinates
    lats = tas_annual.lat.values.astype(np.float64)
    lons = tas_annual.lon.values.astype(np.float64)

    # Ensure lat is ascending (for consistent plotting with origin='lower')
    lat_desc = bool(lats[0] > lats[-1])
    if lat_desc:
        lats = lats[::-1]
        tas_values = tas_values[:, ::-1, :]

    # Extract years from the time coordinate
    years = np.array([int(str(t)[:4]) for t in tas_annual.time.values])

    print(f"  Annual data: {tas_values.shape} — {len(years)} years")
    print(f"  Years: {years[0]} to {years[-1]}")
    print(f"  Lats: {lats[0]:.2f} to {lats[-1]:.2f} ({len(lats)} points)")
    print(f"  Lons: {lons[0]:.2f} to {lons[-1]:.2f} ({len(lons)} points)")
    print(f"  Temp range: {tas_values.min():.1f} to {tas_values.max():.1f} °C")

    # Load precipitation if available; otherwise synthesize it
    pr = None
    if pr_files:
        ds_pr, pr_raw = _open_var(pr_files, "pr")

        if region == "se_asia":
            # Subset to SE Asia (match the tas lat slice direction)
            pr_subset = pr_raw.sel(
                lat=slice(LAT_MAX, LAT_MIN) if lat_vals[0] > lat_vals[-1] else slice(LAT_MIN, LAT_MAX),
                lon=slice(LON_MIN, LON_MAX),
            )
            print(f"  SE Asia subset (pr): {dict(pr_subset.sizes)}")
        else:
            pr_subset = pr_raw
            print(f"  Global (pr): {dict(pr_subset.sizes)}")

        if coarsen > 1:
            print(f"  Coarsening pr by factor {coarsen} (lat/lon)...")
            pr_subset = pr_subset.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

        print("  Resampling daily -> annual mean (pr)...")
        pr_annual = pr_subset.resample(time='1YE').mean()

        # Align to tas annual coordinates (safe even if already aligned)
        tas_annual_aligned, pr_annual_aligned = xr.align(tas_annual, pr_annual, join="inner")
        if tas_annual_aligned.sizes != tas_annual.sizes:
            print(f"  NOTE: aligned time/coords: {dict(tas_annual_aligned.sizes)} (was {dict(tas_annual.sizes)})")
            tas_annual = tas_annual_aligned
            pr_annual = pr_annual_aligned
            tas_values = tas_annual.values - 273.15
            lats = tas_annual.lat.values.astype(np.float64)
            lons = tas_annual.lon.values.astype(np.float64)
            lat_desc = bool(lats[0] > lats[-1])
            if lat_desc:
                lats = lats[::-1]
                tas_values = tas_values[:, ::-1, :]
            years = np.array([int(str(t)[:4]) for t in tas_annual.time.values])

        pr_values = pr_annual.values

        # Convert precipitation units to mm/day if needed.
        pr_units = pr_raw.attrs.get("units", "").lower()
        if ("kg" in pr_units and "s" in pr_units) or ("kg m-2 s-1" in pr_units) or ("kg/m^2/s" in pr_units):
            pr_values = pr_values * 86400.0
            print("  Converted pr from kg m-2 s-1 to mm/day (×86400).")
        else:
            print("  pr unit conversion skipped (units not recognized as kg m-2 s-1).")

        # Flip lat if needed to match tas_values orientation
        if lat_desc:
            pr_values = pr_values[:, ::-1, :]

        pr = pr_values.astype(np.float32)
        print(f"  Precip range: {float(np.nanmin(pr)):.2f} to {float(np.nanmax(pr)):.2f} (mm/day if converted)")

        ds_pr.close()
    else:
        print("  No pr NetCDF files found; generating synthetic precipitation...")
        pr = _synthesize_precipitation(tas_values, lats, lons, years).astype(np.float32)

    # GDP and population (gridded proxies)
    gdp, pop, soil_moisture, coastal_factor = _generate_socioeconomic(lats, lons)

    data = {
        'tas': tas_values.astype(np.float32),
        'pr': pr,
        'gdp': gdp.astype(np.float32),
        'pop': pop.astype(np.float32),
        'soil_moisture': soil_moisture.astype(np.float32),
        'coastal_factor': coastal_factor.astype(np.float32),
        'lats': lats,
        'lons': lons,
        'years': years,
    }

    os.makedirs(out_dir, exist_ok=True)
    cache_name = f"climate_data_{region}_c{int(coarsen)}.pkl"
    pkl_path = os.path.join(out_dir, cache_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved processed data to {pkl_path}")

    ds_tas.close()
    return data


def _synthesize_precipitation(tas, lats, lons, years):
    """
    Generate realistic precipitation from temperature using
    Clausius-Clapeyron scaling and SE Asian monsoon patterns.
    """
    from scipy.ndimage import gaussian_filter

    np.random.seed(42)
    T, nlat, nlon = tas.shape

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    # Base precipitation: higher in tropics and near coast
    coastal_distance = (lon_grid.max() - lon_grid) / (lon_grid.max() - lon_grid.min())
    coastal_factor = np.exp(-3 * coastal_distance)
    pr_base = 6.0 + 3.0 * coastal_factor + 1.5 * np.cos(np.radians(lat_grid) * 2)

    pr = np.zeros_like(tas)
    for t in range(T):
        # Clausius-Clapeyron: +7% per K above baseline
        temp_anomaly = tas[t] - tas[0]
        cc_scaling = 1.0 + 0.07 * np.clip(temp_anomaly, 0, 10)
        var_scale = 1.0 + 0.02 * t

        pr[t] = pr_base * cc_scaling + np.random.normal(0, 1.2 * var_scale, (nlat, nlon))
        corr_rain = gaussian_filter(np.random.normal(0, 1.0, (nlat, nlon)), sigma=3)
        pr[t] += corr_rain
        pr[t] = np.clip(pr[t], 0.01, 30)

    return pr


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
