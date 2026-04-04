"""
Momentum (standardized rate of change) on climate time series.

Implements the momentum formula from Gurjar & Camp (2026):
    m(t) = [lambda(t) - lambda(t-1)] / [sigma_w(t) + epsilon]

This standardizes rate-of-change dimensionlessly across strata.
Positive values indicate acceleration; negative indicate decline.
"""
import numpy as np
from .volatility import _ewma_smooth


def compute_momentum(data_3d, window=3, alpha=0.3, use_ewma=True):
    """
    Standardized momentum at the last timestep (Gurjar & Camp 2026, Eq.2).

        m(t) = [lambda(t) - lambda(t-1)] / [sigma_w(t) + epsilon]

    Input: (T, nlat, nlon)
    Output: (nlat, nlon)
    """
    T = data_3d.shape[0]
    eps = 1e-8

    if use_ewma:
        smoothed = _ewma_smooth(data_3d, alpha=alpha)
    else:
        smoothed = data_3d

    if T < window + 1:
        raw_delta = smoothed[-1] - smoothed[0]
        sigma = np.std(smoothed, axis=0) + eps
        return raw_delta / sigma

    # Raw rate of change
    delta = smoothed[-1] - smoothed[-2]

    # Rolling std for normalization
    sigma = np.std(smoothed[-window:], axis=0) + eps

    # Standardized momentum
    return delta / sigma


def compute_momentum_series(data_3d, window=3, alpha=0.3, use_ewma=True):
    """
    Standardized momentum at every timestep (Gurjar & Camp 2026).

        m(t) = [lambda(t) - lambda(t-1)] / [sigma_w(t) + epsilon]

    Output: (T, nlat, nlon)
    """
    eps = 1e-8

    if use_ewma:
        smoothed = _ewma_smooth(data_3d, alpha=alpha)
    else:
        smoothed = data_3d

    T, nlat, nlon = smoothed.shape
    mom = np.zeros_like(smoothed, dtype=np.float64)

    for t in range(max(1, window), T):
        delta = smoothed[t] - smoothed[t - 1]
        start = max(0, t - window)
        sigma = np.std(smoothed[start:t], axis=0) + eps
        mom[t] = delta / sigma

    # Fill early timesteps
    fill_idx = max(1, window)
    if T > fill_idx:
        mom[:fill_idx] = mom[fill_idx]

    return mom
