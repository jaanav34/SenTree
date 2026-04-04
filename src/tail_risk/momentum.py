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
    deltas = np.diff(data_3d[-window-1:], axis=0)
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
