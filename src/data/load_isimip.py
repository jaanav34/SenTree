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
