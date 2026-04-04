"""
Preprocess climate data into node feature matrices.

Expanded feature set following Gurjar & Camp (2026):
  - EWMA-smoothed intensity signals
  - Standardized momentum
  - Rolling volatility
  - Hawkes self-exciting intensity component
Plus economic features (GDP, population, soil moisture).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def _get_kg_onehot(data, year_idx: int) -> np.ndarray:
    """Get One-Hot KG Climate Classes for a specific year."""
    from src.data.koppen_geiger import classify_grid
    # Ensure raw data is used for classification
    kg_grid = classify_grid(data["tas_monthly"], data["pr_monthly"])[year_idx].flatten()
    
    # One-hot encode (32 classes: 0-31)
    kg_onehot = np.zeros((len(kg_grid), 32), dtype=np.float32)
    for i, val in enumerate(kg_grid):
        if 0 <= val < 32:
            kg_onehot[i, val] = 1.0
    return kg_onehot


def _build_feature_matrix(data, year_idx: int) -> np.ndarray:
    """Build raw feature matrix with One-Hot KG Climate Classes.
    
    Feature layout (43 dims):
        [temp, precip, temp_vol, temp_mom, precip_vol, precip_mom,
         gdp, pop, soil_moisture, coastal_factor, tail_risk_score, 
         ...KG_OneHot(32)]
    """
    tas = data["tas"]
    pr = data["pr"]

    kg_onehot = _get_kg_onehot(data, year_idx)

    temp = tas[year_idx].flatten()
    precip = pr[year_idx].flatten()
    gdp = data["gdp"].flatten()
    pop = data.get("pop", np.ones_like(data["gdp"])).flatten()
    pop_norm = pop / (pop.max() + 1e-8)
    soil = data.get("soil_moisture", np.full_like(data["gdp"], 0.3)).flatten()
    coastal = data.get("coastal_factor", np.zeros_like(data["gdp"])).flatten()

    from src.tail_risk.volatility import compute_volatility
    from src.tail_risk.momentum import compute_momentum
    temp_vol = compute_volatility(tas, window=5, alpha=0.3).flatten()
    temp_mom = compute_momentum(tas, window=3, alpha=0.3).flatten()
    precip_vol = compute_volatility(pr, window=5, alpha=0.3).flatten()
    precip_mom = compute_momentum(pr, window=3, alpha=0.3).flatten()

    from src.tail_risk.engine import get_tail_risk_nodes
    tail_scores, _, _ = get_tail_risk_nodes(data)

    # Combine 11 continuous features + 32 categorical features = 43 dims
    feats = np.column_stack([
        temp, precip, temp_vol, temp_mom, precip_vol, precip_mom,
        gdp, pop_norm, soil, coastal, tail_scores,
        kg_onehot
    ])
    return feats.astype(np.float32)


def _build_positions(data) -> np.ndarray:
    lats, lons = np.meshgrid(data["lats"], data["lons"], indexing="ij")
    return np.column_stack([lats.flatten(), lons.flatten()]).astype(np.float32)


def build_node_features(data, year_idx=-1):
    """
    Build scaled feature matrix for a single timestep.
    Returns: features (N, F), node_positions (N, 2), scaler
    """
    features = _build_feature_matrix(data, year_idx=year_idx)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    positions = _build_positions(data)
    return features.astype(np.float32), positions, scaler


def build_node_features_raw(data, year_idx=-1):
    """Raw (unscaled) features for a single timestep."""
    features = _build_feature_matrix(data, year_idx=year_idx)
    positions = _build_positions(data)
    return features, positions


def build_temporal_features_raw(data):
    """Build raw feature matrices for all timesteps (43 features each)."""
    from src.tail_risk.volatility import compute_volatility_series
    from src.tail_risk.momentum import compute_momentum_series
    from src.tail_risk.engine import get_tail_risk_nodes
    from src.data.koppen_geiger import classify_grid

    tas, pr = data["tas"], data["pr"]
    T, _, _ = tas.shape

    temp_vol_series = compute_volatility_series(tas, window=5, alpha=0.3)
    temp_mom_series = compute_momentum_series(tas, window=3, alpha=0.3)
    precip_vol_series = compute_volatility_series(pr, window=5, alpha=0.3)
    precip_mom_series = compute_momentum_series(pr, window=3, alpha=0.3)

    gdp_flat = data["gdp"].flatten()
    pop_flat = data.get("pop", np.ones_like(data["gdp"])).flatten()
    pop_norm = pop_flat / (pop_flat.max() + 1e-8)
    soil_flat = data.get("soil_moisture", np.full_like(data["gdp"], 0.3)).flatten()
    coastal_flat = data.get("coastal_factor", np.zeros_like(data["gdp"])).flatten()
    tail_scores, _, _ = get_tail_risk_nodes(data)

    # Precompute KG grids for all timesteps
    kg_grids = classify_grid(data["tas_monthly"], data["pr_monthly"])

    features_list = []
    for t in range(T):
        # Get KG one-hot for this timestep
        kg_flat = kg_grids[t].flatten()
        kg_onehot = np.zeros((len(kg_flat), 32), dtype=np.float32)
        for i, val in enumerate(kg_flat):
            if 0 <= val < 32:
                kg_onehot[i, val] = 1.0

        feats = np.column_stack([
            tas[t].flatten(),
            pr[t].flatten(),
            temp_vol_series[t].flatten(),
            temp_mom_series[t].flatten(),
            precip_vol_series[t].flatten(),
            precip_mom_series[t].flatten(),
            gdp_flat,
            pop_norm,
            soil_flat,
            coastal_flat,
            tail_scores,
            kg_onehot
        ])
        features_list.append(feats.astype(np.float32))

    return features_list


def build_temporal_features(data, scaler=None):
    """Build feature matrices for ALL timesteps (optionally scaled)."""
    raw = build_temporal_features_raw(data)
    if scaler is None:
        return raw
    return [scaler.transform(f).astype(np.float32) for f in raw]
