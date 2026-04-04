"""
Tail-Risk Escalation Engine.

Implements the three-regime classification from Gurjar & Camp (2026):
  - Baseline:  low volatility (<0.8), negative/neutral momentum
  - Buildup:   positive momentum (0.2-0.6), moderate volatility (0.2-0.4)
  - Surge:     high volatility (>0.4), strong momentum (>0.6)

Combined with Hawkes-process-inspired self-exciting intensity from
"Estimating high-dimensional Hawkes process with time-dependent covariates"
(Communications in Statistics, 2024).

Binary target (Gurjar & Camp 2026):
    y_t = 1{ max_{tau in [t+1, t+H]} lambda_tau >= q_0.95 }
"""
import numpy as np
from .volatility import compute_volatility, compute_volatility_series, compute_ewma_intensity
from .momentum import compute_momentum, compute_momentum_series


# ---------------------------------------------------------------------------
# Hawkes-process self-exciting intensity (adapted for spatial climate data)
# ---------------------------------------------------------------------------

def _hawkes_intensity(data_3d, mu=None, beta=0.8, decay=0.5):
    """
    Simplified Hawkes process intensity for climate events.

    Conditional intensity:
        lambda*(t) = mu + sum_{t_i < t} beta * exp(-decay * (t - t_i))

    where t_i are timesteps where the signal exceeded a local threshold
    (events = exceedances above the running 90th percentile).

    This captures self-exciting behavior: extreme events beget more extreme events.

    Args:
        data_3d: (T, nlat, nlon) — raw or EWMA-smoothed signal
        mu: (nlat, nlon) — baseline intensity. If None, uses temporal mean.
        beta: excitation magnitude per event
        decay: exponential decay rate

    Returns:
        intensity: (T, nlat, nlon) — Hawkes intensity at each timestep
    """
    T, nlat, nlon = data_3d.shape
    intensity = np.zeros_like(data_3d, dtype=np.float64)

    if mu is None:
        mu = np.mean(data_3d, axis=0)

    # Define events as exceedances above running 90th percentile
    running_q90 = np.zeros((nlat, nlon), dtype=np.float64)
    running_q90[:] = np.percentile(data_3d[:max(5, T // 4)], 90, axis=0)

    intensity[0] = mu.copy()

    for t in range(1, T):
        # Decay all previous excitations
        intensity[t] = mu.copy()

        # Sum excitation from all past events with exponential decay
        for t_i in range(max(0, t - 10), t):  # lookback window of 10
            # Event mask: signal exceeded the threshold at time t_i
            event_mask = data_3d[t_i] > running_q90
            excitation = beta * np.exp(-decay * (t - t_i))
            intensity[t] += event_mask * excitation

        # Update running threshold
        start = max(0, t - 10)
        running_q90 = np.percentile(data_3d[start:t + 1], 90, axis=0)

    return intensity


def _classify_regime(volatility, momentum):
    """
    Three-regime classification (Gurjar & Camp 2026, Section 4).

    Returns:
        regime: (nlat, nlon) — 0=Baseline, 1=Buildup, 2=Surge
    """
    regime = np.zeros_like(volatility, dtype=np.int32)

    # Normalize to [0, 1] for regime boundaries
    vol_n = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
    mom_n = (momentum - momentum.min()) / (momentum.max() - momentum.min() + 1e-8)

    # Buildup: positive momentum (0.2-0.6), moderate volatility (0.2-0.4)
    buildup_mask = (mom_n > 0.2) & (mom_n < 0.6) & (vol_n > 0.2) & (vol_n < 0.6)
    regime[buildup_mask] = 1

    # Surge: high volatility (>0.4), strong momentum (>0.6)
    surge_mask = (vol_n > 0.4) & (mom_n > 0.6)
    regime[surge_mask] = 2

    return regime


def compute_tail_risk(data, vol_weight=0.35, mom_weight=0.35, hawkes_weight=0.30,
                      percentile=95, alpha=0.3):
    """
    Compute tail_risk_score for each grid cell using the full Gurjar & Camp (2026)
    framework plus Hawkes self-exciting intensity.

    Feature vector per node: x_t = (lambda_t, m_t, v_t) + Hawkes intensity

    Returns:
        scores: (nlat, nlon) — composite tail risk score
        flags: (nlat, nlon) — boolean, True if above threshold
        threshold: float — the 95th percentile cutoff
        regime: (nlat, nlon) — regime classification (0/1/2)
        components: dict — individual score components for analysis
    """
    tas = data['tas']
    pr = data['pr']

    # EWMA-smoothed intensity (Gurjar & Camp 2026, Eq.1)
    temp_intensity = compute_ewma_intensity(tas, alpha=alpha)
    precip_intensity = compute_ewma_intensity(pr, alpha=alpha)

    # Volatility (Gurjar & Camp 2026, Eq.3)
    temp_vol = compute_volatility(tas, window=5, alpha=alpha)
    precip_vol = compute_volatility(pr, window=5, alpha=alpha)

    # Standardized momentum (Gurjar & Camp 2026, Eq.2)
    temp_mom = compute_momentum(tas, window=3, alpha=alpha)
    precip_mom = compute_momentum(pr, window=3, alpha=alpha)

    # Hawkes self-exciting intensity
    temp_hawkes = _hawkes_intensity(tas)[-1]    # last timestep
    precip_hawkes = _hawkes_intensity(pr)[-1]

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    # Multi-signal composite score
    vol_component = (norm(temp_vol) + norm(precip_vol)) / 2
    mom_component = (norm(temp_mom) + norm(precip_mom)) / 2
    hawkes_component = (norm(temp_hawkes) + norm(precip_hawkes)) / 2

    score = (
        vol_weight * vol_component +
        mom_weight * mom_component +
        hawkes_weight * hawkes_component
    )

    threshold = np.percentile(score, percentile)
    flags = score >= threshold

    # Regime classification
    combined_vol = (temp_vol + precip_vol) / 2
    combined_mom = (temp_mom + precip_mom) / 2
    regime = _classify_regime(combined_vol, combined_mom)

    components = {
        'volatility': vol_component,
        'momentum': mom_component,
        'hawkes': hawkes_component,
        'temp_vol': temp_vol,
        'precip_vol': precip_vol,
        'temp_mom': temp_mom,
        'precip_mom': precip_mom,
        'regime': regime,
    }

    return score, flags, threshold, regime, components


def compute_tail_risk_series(data, window=5, alpha=0.3, percentile=95,
                              vol_weight=0.35, mom_weight=0.35, hawkes_weight=0.30):
    """
    Compute tail-risk scores and flags for every timestep.

    Returns:
        scores_series: list of (nlat, nlon) arrays
        flags_series: list of (nlat, nlon) boolean arrays
        regime_series: list of (nlat, nlon) int arrays
    """
    tas, pr = data['tas'], data['pr']
    T = tas.shape[0]

    temp_vol_s = compute_volatility_series(tas, window=window, alpha=alpha)
    precip_vol_s = compute_volatility_series(pr, window=window, alpha=alpha)
    temp_mom_s = compute_momentum_series(tas, window=3, alpha=alpha)
    precip_mom_s = compute_momentum_series(pr, window=3, alpha=alpha)

    # Hawkes intensity series
    temp_hawkes_s = _hawkes_intensity(tas)
    precip_hawkes_s = _hawkes_intensity(pr)

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    scores_series = []
    flags_series = []
    regime_series = []

    for t in range(T):
        vol_c = (norm(temp_vol_s[t]) + norm(precip_vol_s[t])) / 2
        mom_c = (norm(temp_mom_s[t]) + norm(precip_mom_s[t])) / 2
        haw_c = (norm(temp_hawkes_s[t]) + norm(precip_hawkes_s[t])) / 2

        score = vol_weight * vol_c + mom_weight * mom_c + hawkes_weight * haw_c
        threshold = np.percentile(score, percentile)

        scores_series.append(score)
        flags_series.append(score >= threshold)

        combined_vol = (temp_vol_s[t] + precip_vol_s[t]) / 2
        combined_mom = (temp_mom_s[t] + precip_mom_s[t]) / 2
        regime_series.append(_classify_regime(combined_vol, combined_mom))

    return scores_series, flags_series, regime_series


def get_tail_risk_nodes(data):
    """Returns flat arrays for graph construction."""
    scores, flags, threshold, regime, components = compute_tail_risk(data)
    return scores.flatten(), flags.flatten(), threshold
