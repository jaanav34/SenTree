"""Tail-Risk Escalation Engine — combines volatility + momentum, flags 95th percentile."""
import numpy as np
from .volatility import compute_volatility
from .momentum import compute_momentum


def compute_tail_risk(data, vol_weight=0.6, mom_weight=0.4, percentile=95):
    """
    Compute tail_risk_score for each grid cell.

    Returns:
        scores: (nlat, nlon) — composite tail risk score
        flags: (nlat, nlon) — boolean, True if above threshold
        threshold: float — the 95th percentile cutoff
    """
    tas = data['tas']
    pr = data['pr']

    temp_vol = compute_volatility(tas, window=5)
    temp_mom = compute_momentum(tas, window=3)
    precip_vol = compute_volatility(pr, window=5)
    precip_mom = compute_momentum(pr, window=3)

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    score = (
        vol_weight * (norm(temp_vol) + norm(precip_vol)) / 2 +
        mom_weight * (norm(temp_mom) + norm(precip_mom)) / 2
    )

    threshold = np.percentile(score, percentile)
    flags = score >= threshold

    return score, flags, threshold


def get_tail_risk_nodes(data):
    """Returns flat arrays for graph construction."""
    scores, flags, threshold = compute_tail_risk(data)
    return scores.flatten(), flags.flatten(), threshold
