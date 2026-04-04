def _load_from_netcdf(
    tas_files,
    pr_files,
    out_dir,
    *,
    region: str,
    coarsen: int,
    out_pkl_path: str,
):
    """
    Load ISIMIP3b daily NetCDF files, resample to both annual and monthly,
    and align/convert units.
    """
    import xarray as xr

    def _open_var(files, varname: str):
        # ... (Keep the existing _open_var logic from your file) ...
        try:
            import importlib.util
            chunks = {"time": 365} if importlib.util.find_spec("dask") else None
        except Exception:
            chunks = None

        ds = xr.open_mfdataset(files, combine='by_coords', chunks=chunks)
        return ds, ds[varname]

    # 1. LOAD TEMPERATURE (TAS)
    ds_tas, tas_raw = _open_var(tas_files, "tas")
    lat_vals = tas_raw.lat.values

    # Subset & Coarsen
    if region == "se_asia":
        lat_slice = slice(LAT_MAX, LAT_MIN) if lat_vals[0] > lat_vals[-1] else slice(LAT_MIN, LAT_MAX)
        tas_subset = tas_raw.sel(lat=lat_slice, lon=slice(LON_MIN, LON_MAX))
    else:
        tas_subset = tas_raw

    if coarsen > 1:
        tas_subset = tas_subset.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

    # Resample TAS: Both Annual and Monthly
    tas_annual = tas_subset.resample(time='1YE').mean()
    tas_monthly = tas_subset.resample(time='1MS').mean()

    # 2. LOAD PRECIPITATION (PR)
    pr_annual, pr_monthly = None, None

    if pr_files:
        ds_pr, pr_raw = _open_var(pr_files, "pr")
        
        # Subset & Coarsen PR to match TAS
        if region == "se_asia":
            pr_subset = pr_raw.sel(
                lat=slice(LAT_MAX, LAT_MIN) if lat_vals[0] > lat_vals[-1] else slice(LAT_MIN, LAT_MAX),
                lon=slice(LON_MIN, LON_MAX),
            )
        else:
            pr_subset = pr_raw

        if coarsen > 1:
            pr_subset = pr_subset.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

        # Resample PR: Both Annual and Monthly
        pr_annual = pr_subset.resample(time='1YE').mean()
        pr_monthly = pr_subset.resample(time='1MS').mean()

        # Align time coordinates (ensure we have the same years/months)
        tas_annual, pr_annual = xr.align(tas_annual, pr_annual, join="inner")
        tas_monthly, pr_monthly = xr.align(tas_monthly, pr_monthly, join="inner")
        
        # Unit Conversion for PR (kg m-2 s-1 -> mm/day)
        pr_units = pr_raw.attrs.get("units", "").lower()
        if any(x in pr_units for x in ["kg", "s"]):
            pr_annual_values = pr_annual.values * 86400.0
            pr_monthly_values = pr_monthly.values * 86400.0
        else:
            pr_annual_values = pr_annual.values
            pr_monthly_values = pr_monthly.values
    else:
        # If no real PR, we will synthesize later using the converted TAS
        pr_annual_values, pr_monthly_values = None, None

    # 3. PREPARE FINAL ARRAYS
    years = np.array([int(str(t)[:4]) for t in tas_annual.time.values])
    T_years = len(years)
    lats = tas_annual.lat.values.astype(np.float64)
    lons = tas_annual.lon.values.astype(np.float64)

    # Convert TAS K -> °C
    tas_annual_values = tas_annual.values - 273.15
    tas_monthly_values = tas_monthly.values - 273.15

    # Handle Descending Latitudes (Flip for consistent plotting)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        tas_annual_values = tas_annual_values[:, ::-1, :]
        tas_monthly_values = tas_monthly_values[:, ::-1, :]
        if pr_annual_values is not None:
            pr_annual_values = pr_annual_values[:, ::-1, :]
            pr_monthly_values = pr_monthly_values[:, ::-1, :]

    # 4. SYNTHESIZE PR IF MISSING
    if pr_annual_values is None:
        print("  Generating synthetic precipitation (monthly + annual)...")
        # Reshape TAS monthly for synthesis: (Years, 12, Lat, Lon)
        nlat, nlon = len(lats), len(lons)
        tas_m_reshape = tas_monthly_values[:T_years*12].reshape(T_years, 12, nlat, nlon)
        pr_monthly_values = _synthesize_monthly_precipitation(tas_m_reshape, lats, lons, years)
        pr_annual_values = np.mean(pr_monthly_values, axis=1)
    else:
        # Reshape real monthly PR to (Years, 12, Lat, Lon)
        nlat, nlon = len(lats), len(lons)
        pr_monthly_values = pr_monthly_values[:T_years*12].reshape(T_years, 12, nlat, nlon)

    # Reshape TAS monthly for final dict
    tas_monthly_values = tas_monthly_values[:T_years*12].reshape(T_years, 12, nlat, nlon)

    # 5. SOCIOECONOMIC & CACHING
    gdp, pop, soil, coastal = _generate_socioeconomic(lats, lons)

    data = {
        'tas': tas_annual_values.astype(np.float32),
        'tas_monthly': tas_monthly_values.astype(np.float32),
        'pr': pr_annual_values.astype(np.float32),
        'pr_monthly': pr_monthly_values.astype(np.float32),
        'gdp': gdp.astype(np.float32),
        'pop': pop.astype(np.float32),
        'soil_moisture': soil.astype(np.float32),
        'coastal_factor': coastal.astype(np.float32),
        'lats': lats,
        'lons': lons,
        'years': years,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(data, f)
    
    ds_tas.close()
    if pr_files: ds_pr.close()
    return data