"""Rolling volatility (standard deviation) on climate time series."""
import numpy as np


def compute_volatility(data_3d, window=5):
    """
    Rolling std over time axis.
    Input: (T, nlat, nlon)
    Output: (nlat, nlon) — volatility at the last valid timestep
    """
    T = data_3d.shape[0]
    if T < window:
        return np.std(data_3d, axis=0)
    return np.std(data_3d[-window:], axis=0)


def compute_volatility_series(data_3d, window=5):
    """
    Rolling std for every timestep.
    Output: (T, nlat, nlon)
    """
    T, nlat, nlon = data_3d.shape
    vol = np.zeros_like(data_3d)
    for t in range(window, T):
        vol[t] = np.std(data_3d[t-window:t], axis=0)
    if T > window:
        vol[:window] = vol[window]
    return vol
