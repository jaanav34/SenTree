"""
Preprocess climate data into node feature matrices.

Expanded feature set following Gurjar & Camp (2026):
  - EWMA-smoothed intensity signals
  - Standardized momentum
  - Rolling volatility
  - Hawkes self-exciting intensity component
Plus economic features (GDP, population, soil moisture).

Key fixes (v2):
  1. tail_risk_score is now a TRUE TIME-SERIES feature computed per
     timestep via compute_tail_risk_series, not a static snapshot
     stamped identically across all T frames. The old approach made
     feature[10] a constant column, causing the GNN to see identical
     "risk context" regardless of year while vol/mom changed — the
     mismatch drove the spike pattern.

  2. StandardScaler is now fitted on ALL timesteps stacked together,
     not just year_idx=-1. Fitting on a single frame means the scaler's
     mean/std reflects only that snapshot; when applied to other
     timesteps the tail_risk and vol columns land in completely
     different z-score ranges, amplifying inter-timestep variance.

  3. KG one-hot encoding vectorized (no Python loop over nodes).

  4. _build_feature_matrix removed: it was only used by build_node_features
     and build_node_features_raw, both of which now delegate to the
     temporal pipeline for consistency.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kg_onehot_vectorized(kg_flat: np.ndarray, n_classes: int = 32) -> np.ndarray:
    """Vectorized one-hot encoding — no Python loop over nodes."""
    N = len(kg_flat)
    onehot = np.zeros((N, n_classes), dtype=np.float32)
    valid = (kg_flat >= 0) & (kg_flat < n_classes)
    onehot[np.where(valid), kg_flat[valid]] = 1.0
    return onehot


def _build_positions(data) -> np.ndarray:
    lats, lons = np.meshgrid(data["lats"], data["lons"], indexing="ij")
    return np.column_stack([lats.flatten(), lons.flatten()]).astype(np.float32)


def _precompute_series(data):
    """
    Compute all time-varying feature series once and return as a dict.
    Called internally by build_temporal_features_raw so that every
    public function shares the same computation path.
    """
    from src.tail_risk.volatility import compute_volatility_series
    from src.tail_risk.momentum import compute_momentum_series
    from src.tail_risk.engine import compute_tail_risk_series
    from src.data.koppen_geiger import classify_grid

    tas, pr = data["tas"], data["pr"]

    temp_vol_series    = compute_volatility_series(tas, window=5, alpha=0.3)   # (T,nlat,nlon)
    temp_mom_series    = compute_momentum_series(tas,   window=3, alpha=0.3)
    precip_vol_series  = compute_volatility_series(pr,  window=5, alpha=0.3)
    precip_mom_series  = compute_momentum_series(pr,    window=3, alpha=0.3)

    # FIX 1: per-timestep tail-risk scores from the series engine
    # smoothed_scores is a list of T arrays each (nlat, nlon), already
    # globally rescaled to [0,1] inside compute_tail_risk_series.
    smoothed_scores, _, _ = compute_tail_risk_series(data)
    tail_risk_series = np.stack(smoothed_scores, axis=0)   # (T, nlat, nlon)

    # KG grids (T, nlat, nlon) integer class labels
    kg_grids = None
    existing_kg = data.get("kg_codes")
    if isinstance(existing_kg, np.ndarray):
        existing_kg = np.asarray(existing_kg)
        if existing_kg.shape[:1] == (T,):
            kg_grids = existing_kg.astype(np.int32, copy=False)

    if kg_grids is None:
        kg_grids = classify_grid(data["tas_monthly"], data["pr_monthly"])
        data["kg_codes"] = kg_grids

    return {
        "temp_vol":   temp_vol_series,
        "temp_mom":   temp_mom_series,
        "precip_vol": precip_vol_series,
        "precip_mom": precip_mom_series,
        "tail_risk":  tail_risk_series,
        "kg_grids":   kg_grids,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_temporal_features_raw(data):
    """
    Build raw (unscaled) feature matrices for ALL timesteps.

    Returns: list of T arrays, each (N, 43)
      Feature layout:
        [0]  temp            [1]  precip
        [2]  temp_vol        [3]  temp_mom
        [4]  precip_vol      [5]  precip_mom
        [6]  gdp             [7]  pop_norm
        [8]  soil_moisture   [9]  coastal_factor
        [10] tail_risk_score (per-timestep)
        [11..42] KG one-hot (32 dims)
    """
    tas, pr = data["tas"], data["pr"]
    T = tas.shape[0]

    series = _precompute_series(data)

    gdp_flat    = data["gdp"].flatten()
    pop_flat    = data.get("pop", np.ones_like(data["gdp"])).flatten()
    pop_norm    = (pop_flat / (pop_flat.max() + 1e-8)).astype(np.float32)
    soil_flat   = data.get("soil_moisture",  np.full_like(data["gdp"], 0.3)).flatten()
    coastal_flat = data.get("coastal_factor", np.zeros_like(data["gdp"])).flatten()

    features_list = []
    for t in range(T):
        kg_onehot = _kg_onehot_vectorized(series["kg_grids"][t].flatten())

        feats = np.column_stack([
            tas[t].flatten(),                          # [0]  temp
            pr[t].flatten(),                           # [1]  precip
            series["temp_vol"][t].flatten(),           # [2]  temp_vol
            series["temp_mom"][t].flatten(),           # [3]  temp_mom
            series["precip_vol"][t].flatten(),         # [4]  precip_vol
            series["precip_mom"][t].flatten(),         # [5]  precip_mom
            gdp_flat,                                  # [6]  gdp
            pop_norm,                                  # [7]  pop_norm
            soil_flat,                                 # [8]  soil_moisture
            coastal_flat,                              # [9]  coastal_factor
            series["tail_risk"][t].flatten(),          # [10] tail_risk (per-t)
            kg_onehot,                                 # [11..42] KG one-hot
        ])
        features_list.append(feats.astype(np.float32))

    return features_list


def build_temporal_features(data, scaler=None):
    """
    Build scaled feature matrices for ALL timesteps.

    FIX 2: If no scaler is provided, we fit one on ALL timesteps stacked
    together so that the mean/std reflects the full temporal distribution,
    not just a single snapshot. This prevents the scaler from mapping
    features from other years into wildly different z-score ranges.
    """
    raw = build_temporal_features_raw(data)

    if scaler is None:
        # Fit on the full temporal distribution
        all_frames = np.vstack(raw)           # (T*N, 43)
        scaler = StandardScaler()
        scaler.fit(all_frames)

    return [scaler.transform(f).astype(np.float32) for f in raw], scaler


def build_node_features(data, year_idx=-1):
    """
    Build scaled feature matrix for a single timestep.

    Delegates to the temporal pipeline so that the scaler is always
    fitted on the full distribution, then extracts the requested year.

    Returns: features (N, 43), node_positions (N, 2), scaler
    """
    scaled_list, scaler = build_temporal_features(data, scaler=None)
    features = scaled_list[year_idx]
    positions = _build_positions(data)
    return features, positions, scaler


def build_node_features_raw(data, year_idx=-1):
    """Raw (unscaled) features for a single timestep."""
    raw = build_temporal_features_raw(data)
    positions = _build_positions(data)
    return raw[year_idx], positions
