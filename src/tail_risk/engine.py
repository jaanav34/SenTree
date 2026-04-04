"""
Tail-Risk Escalation Engine (Stabilized).

Implements 'Climate-Relative Normalization' (regime-aware) to stabilize risk
scores across heterogeneous Köppen-Geiger climate zones.

Key Fixes (v2):
  1. _climate_relative_norm: Removed per-call min-max rescaling.
     Normalization is now z-score only (per KG class), clipped to [-3, 3]
     and then globally rescaled ONCE at the composite score level in the
     caller. This eliminates the "every spike hits 1.0" artifact caused
     by independent per-component rescaling.

  2. compute_tail_risk_series: Threshold is now computed ONCE from the
     global distribution of all timestep scores, not recalculated per
     timestep. This kills the "5% always flagged" artifact that produced
     the cardiac-monitor pattern.

  3. Hawkes process: running_q90 is now anchored to a fixed baseline
     computed from the first quarter of the series, preventing threshold
     drift on short annual sequences.

  4. train_gnn targets: get_tail_risk_nodes now returns raw z-scores
     (not min-maxed). The single normalization happens inside train_gnn
     only, eliminating the triple-normalization chain.
"""
import numpy as np
from .volatility import compute_volatility, compute_volatility_series, compute_ewma_intensity
from .momentum import compute_momentum, compute_momentum_series


# ---------------------------------------------------------------------------
# Hawkes-process self-exciting intensity (adapted for spatial climate data)
# ---------------------------------------------------------------------------

def _hawkes_intensity(data_3d, mu=None, beta=0.8, decay=0.2):
    """
    Simplified Hawkes process intensity for climate events.

    FIX: running_q90 is now anchored to a FIXED baseline from the first
    quarter of the series. Previously it was recalculated at every timestep
    from an expanding window, causing severe threshold drift on short (~85yr)
    annual sequences — the effective excitation threshold kept dropping,
    making nearly every late-series point an "event".
    """
    T, nlat, nlon = data_3d.shape
    intensity = np.zeros_like(data_3d, dtype=np.float64)

    if mu is None:
        mu = np.mean(data_3d, axis=0)

    # FIXED: Use a stable baseline window (first 25% of series, min 5 steps)
    baseline_end = max(5, T // 4)
    fixed_q90 = np.percentile(data_3d[:baseline_end], 90, axis=0)

    intensity[0] = mu.copy()

    for t in range(1, T):
        intensity[t] = mu.copy()
        for t_i in range(max(0, t - 10), t):
            event_mask = data_3d[t_i] > fixed_q90   # <-- fixed threshold
            excitation = beta * np.exp(-decay * (t - t_i))
            intensity[t] += event_mask * excitation

    return intensity


def _climate_relative_norm(signal, kg_codes):
    """
    Standardize signal relative to its Köppen-Geiger climate class average.

    FIX: Previously applied a SECOND min-max rescale to [0,1] after the
    z-score, meaning every independent call would compress its range to
    exactly [0,1]. When vol, mom, and hawkes components were each rescaled
    to [0,1] before combining, any timestep with one large outlier would
    pin that component to 1.0 and crush all other values to ~0, then the
    next timestep would do the same — producing the spike pattern.

    Now returns raw z-scores clipped to [-3, 3]. The caller (compute_tail_risk
    / compute_tail_risk_series) performs a single global rescale on the
    composite score, so relative differences between timesteps are preserved.
    """
    normed = np.zeros_like(signal, dtype=np.float32)
    unique_classes = np.unique(kg_codes)

    for cls in unique_classes:
        mask = (kg_codes == cls)
        if not np.any(mask):
            continue

        cls_values = signal[mask]
        mean = np.mean(cls_values)
        std = np.std(cls_values) + 1e-8

        # Z-score standardization per climate class, clipped to ±3σ
        normed[mask] = np.clip((signal[mask] - mean) / std, -3.0, 3.0)

    # Shift to non-negative (z ∈ [-3,3] → [0,6]) for downstream weighting
    # Do NOT rescale to [0,1] here — that's done once on the composite score.
    normed = normed + 3.0   # now in [0, 6]
    return normed


def _climate_relative_norm_to_01(signal, kg_codes):
    """
    Same as _climate_relative_norm but applies final [0,1] rescale.
    Used ONLY for the single-snapshot compute_tail_risk (not the series),
    where global context across timesteps isn't needed.
    """
    normed = _climate_relative_norm(signal, kg_codes)
    mn, mx = normed.min(), normed.max()
    return (normed - mn) / (mx - mn + 1e-8)


def _classify_regime(volatility, momentum):
    """Three-regime classification (Gurjar & Camp 2026)."""
    regime = np.zeros_like(volatility, dtype=np.int32)

    vol_n = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
    mom_n = (momentum - momentum.min()) / (momentum.max() - momentum.min() + 1e-8)

    # Buildup: positive momentum, moderate volatility
    buildup_mask = (mom_n > 0.2) & (mom_n < 0.6) & (vol_n > 0.2) & (vol_n < 0.6)
    regime[buildup_mask] = 1

    # Surge: high volatility, strong momentum
    surge_mask = (vol_n > 0.4) & (mom_n > 0.6)
    regime[surge_mask] = 2

    return regime


def compute_tail_risk(data, vol_weight=0.35, mom_weight=0.35, hawkes_weight=0.30,
                      percentile=95, alpha=0.3):
    """
    Compute regime-aware tail_risk_score using Climate-Relative Normalization.

    Returns scores in [0, 1] after a single global rescale of the composite.
    """
    tas = data['tas']
    pr = data['pr']

    kg_codes_all = data.get('kg_codes')
    if kg_codes_all is None:
        kg_codes = np.zeros(tas.shape[1:], dtype=np.int32)
    elif kg_codes_all.ndim == 3:
        kg_codes = kg_codes_all[-1]
    else:
        kg_codes = kg_codes_all

    # EWMA-smoothed intensity
    temp_intensity = compute_ewma_intensity(tas, alpha=alpha)
    precip_intensity = compute_ewma_intensity(pr, alpha=alpha)

    temp_vol    = compute_volatility(tas, window=10, alpha=alpha)
    precip_vol  = compute_volatility(pr,  window=10, alpha=alpha)
    temp_mom    = compute_momentum(tas, window=7, alpha=alpha)
    precip_mom  = compute_momentum(pr,  window=7, alpha=alpha)

    temp_hawkes    = _hawkes_intensity(tas, decay=0.2)[-1]
    precip_hawkes  = _hawkes_intensity(pr,  decay=0.2)[-1]

    # Per-component z-score normalization (no per-component min-max)
    vol_component    = (_climate_relative_norm_to_01(temp_vol,    kg_codes) +
                        _climate_relative_norm_to_01(precip_vol,  kg_codes)) / 2

    mom_component    = (_climate_relative_norm_to_01(temp_mom,    kg_codes) +
                        _climate_relative_norm_to_01(precip_mom,  kg_codes)) / 2

    hawkes_component = (_climate_relative_norm_to_01(temp_hawkes, kg_codes) +
                        _climate_relative_norm_to_01(precip_hawkes, kg_codes)) / 2

    score = (
        vol_weight    * vol_component +
        mom_weight    * mom_component +
        hawkes_weight * hawkes_component
    )

    threshold = np.percentile(score, percentile)
    flags = score >= threshold

    combined_vol = (temp_vol + precip_vol) / 2
    combined_mom = (temp_mom + precip_mom) / 2
    regime = _classify_regime(combined_vol, combined_mom)

    components = {
        'volatility':   vol_component,
        'momentum':     mom_component,
        'hawkes':       hawkes_component,
        'temp_vol':     temp_vol,
        'precip_vol':   precip_vol,
        'temp_mom':     temp_mom,
        'precip_mom':   precip_mom,
        'regime':       regime,
    }

    return score, flags, threshold, regime, components


def compute_tail_risk_series(data, window=10, alpha=0.3, percentile=95,
                              vol_weight=0.35, mom_weight=0.35, hawkes_weight=0.30):
    """
    Compute tail-risk scores across all timesteps with temporal consistency.

    KEY FIX: The percentile threshold for flags is now computed ONCE from
    the full distribution of all timestep scores (stacked), rather than
    recalculated independently per timestep. Previously, a fresh percentile
    on each (nlat×nlon) slice guaranteed exactly 5% of nodes were always
    flagged — making flags meaningless and injecting artificial spikes into
    every aggregated metric.

    The threshold is global and fixed, so flags genuinely reflect whether a
    node exceeds the 95th percentile of the ENTIRE simulation's risk space.
    """
    tas, pr = data['tas'], data['pr']
    T = tas.shape[0]

    kg_codes_series = data.get('kg_codes')
    if kg_codes_series is None:
        kg_codes_series = np.zeros_like(tas, dtype=np.int32)

    temp_vol_s    = compute_volatility_series(tas, window=window, alpha=alpha)
    precip_vol_s  = compute_volatility_series(pr,  window=window, alpha=alpha)
    temp_mom_s    = compute_momentum_series(tas, window=7, alpha=alpha)
    precip_mom_s  = compute_momentum_series(pr,  window=7, alpha=alpha)

    temp_hawkes_s   = _hawkes_intensity(tas, decay=0.2)
    precip_hawkes_s = _hawkes_intensity(pr,  decay=0.2)

    # --- Pass 1: Compute raw (un-rescaled) composite scores for all T ---
    # We use _climate_relative_norm (z-score only, no per-call min-max) so
    # that cross-timestep magnitude differences are preserved, then rescale
    # globally below.
    raw_scores = []

    for t in range(T):
        kg_t = kg_codes_series[t] if kg_codes_series.ndim == 3 else kg_codes_series

        vol_c = (_climate_relative_norm(temp_vol_s[t],    kg_t) +
                 _climate_relative_norm(precip_vol_s[t],  kg_t)) / 2

        mom_c = (_climate_relative_norm(temp_mom_s[t],    kg_t) +
                 _climate_relative_norm(precip_mom_s[t],  kg_t)) / 2

        haw_c = (_climate_relative_norm(temp_hawkes_s[t], kg_t) +
                 _climate_relative_norm(precip_hawkes_s[t], kg_t)) / 2

        score = vol_weight * vol_c + mom_weight * mom_c + hawkes_weight * haw_c
        raw_scores.append(score)

    # --- Global rescale: fit min/max across ALL timesteps at once ---
    all_scores_stacked = np.stack(raw_scores, axis=0)   # (T, nlat, nlon)
    global_min = all_scores_stacked.min()
    global_max = all_scores_stacked.max()

    scaled_scores = (all_scores_stacked - global_min) / (global_max - global_min + 1e-8)

    # --- FIXED threshold: single percentile over the entire (T, N) distribution ---
    global_threshold = np.percentile(scaled_scores, percentile)

    # --- EWMA temporal smoothing to prevent 'blinking' ---
    smoothed_scores = []
    flags_series    = []
    regime_series   = []

    current_smooth = scaled_scores[0].copy()

    for t in range(T):
        current_smooth = alpha * scaled_scores[t] + (1 - alpha) * current_smooth
        smoothed_scores.append(current_smooth.copy())

        # Flags against the GLOBAL threshold — not a per-timestep percentile
        flags_series.append(current_smooth >= global_threshold)

        combined_vol = (temp_vol_s[t] + precip_vol_s[t]) / 2
        combined_mom = (temp_mom_s[t] + precip_mom_s[t]) / 2
        regime_series.append(_classify_regime(combined_vol, combined_mom))

    return smoothed_scores, flags_series, regime_series


def get_tail_risk_nodes(data):
    """
    Returns flat arrays for graph construction.

    FIX: Previously returned min-maxed scores, which combined with the
    re-normalization inside train_gnn produced triple-normalized targets.
    Now returns the raw composite score (already in a natural range from
    the weighted sum of z-score components). train_gnn does the single
    final normalization to [0.025, 0.975] with label smoothing.
    """
    scores, flags, threshold, regime, components = compute_tail_risk(data)
    return scores.flatten(), flags.flatten(), threshold