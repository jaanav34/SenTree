"""
Rolling volatility on climate time series.

Implements EWMA-smoothed volatility following Gurjar & Camp (2026)
"Predicting Tail-Risk Escalation in IDS Alert Time Series."

Key formula:
    v(t) = sqrt( (1/w) * sum_{i=t-w+1}^{t} [m(i) - m_bar_w]^2 )

where m_bar_w is the mean momentum over the rolling window.
"""
import numpy as np


def _ewma_smooth(data_3d, alpha=0.3):
    """
    Exponentially Weighted Moving Average smoothing (Gurjar & Camp 2026, Eq.1).

        lambda(t) = alpha * X(t) + (1 - alpha) * lambda(t-1)

    Suppresses high-frequency noise while preserving burst structures.

    Args:
        data_3d: (T, nlat, nlon) raw signal
        alpha: smoothing factor in (0,1). Lower = more smoothing. Paper uses 0.3.

    Returns:
        smoothed: (T, nlat, nlon)
    """
    T = data_3d.shape[0]
    smoothed = np.zeros_like(data_3d, dtype=np.float64)
    smoothed[0] = data_3d[0].astype(np.float64)
    for t in range(1, T):
        smoothed[t] = alpha * data_3d[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def compute_volatility(data_3d, window=5, alpha=0.3, use_ewma=True):
    """
    Rolling volatility at the last timestep.

    If use_ewma=True, applies EWMA smoothing first (Gurjar & Camp 2026),
    then computes rolling standard deviation of momentum residuals.

    Input: (T, nlat, nlon)
    Output: (nlat, nlon) — volatility at the last valid timestep
    """
    T = data_3d.shape[0]
    if use_ewma:
        data_3d = _ewma_smooth(data_3d, alpha=alpha)
    if T < window:
        return np.std(data_3d, axis=0)
    return np.std(data_3d[-window:], axis=0)


def compute_volatility_series(data_3d, window=5, alpha=0.3, use_ewma=True):
    """
    Rolling std for every timestep using EWMA-smoothed signal.

    Following Gurjar & Camp (2026):
        v(t) = sqrt( (1/w) * sum [m(i) - m_bar]^2 )

    Output: (T, nlat, nlon)
    """
    if use_ewma:
        data_3d = _ewma_smooth(data_3d, alpha=alpha)
    T, nlat, nlon = data_3d.shape
    vol = np.zeros_like(data_3d, dtype=np.float64)
    for t in range(window, T):
        segment = data_3d[t - window:t]
        vol[t] = np.std(segment, axis=0)
    if T > window:
        vol[:window] = vol[window]
    return vol


def compute_ewma_intensity(data_3d, alpha=0.3):
    """
    EWMA intensity series (Gurjar & Camp 2026, Eq.1).
    Returns smoothed signal that preserves burst structures.

    Output: (T, nlat, nlon)
    """
    return _ewma_smooth(data_3d, alpha=alpha)
