"""Preprocess climate data into node feature matrices."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def _build_feature_matrix(data, year_idx: int) -> np.ndarray:
    tas = data["tas"]  # (T, nlat, nlon)
    pr = data["pr"]

    temp = tas[year_idx].flatten()
    precip = pr[year_idx].flatten()
    gdp = data["gdp"].flatten()

    from src.tail_risk.volatility import compute_volatility
    from src.tail_risk.momentum import compute_momentum

    temp_vol = compute_volatility(tas, window=5).flatten()
    temp_mom = compute_momentum(tas, window=3).flatten()
    precip_vol = compute_volatility(pr, window=5).flatten()
    precip_mom = compute_momentum(pr, window=3).flatten()

    # Features:
    # [temp, precip, temp_vol, temp_mom, precip_vol, precip_mom, gdp]
    feats = np.column_stack([temp, precip, temp_vol, temp_mom, precip_vol, precip_mom, gdp])
    return feats.astype(np.float32)


def _build_positions(data) -> np.ndarray:
    lats, lons = np.meshgrid(data["lats"], data["lons"], indexing="ij")
    return np.column_stack([lats.flatten(), lons.flatten()]).astype(np.float32)


def build_node_features(data, year_idx=-1):
    """
    Build feature matrix for a single timestep.
    Returns: features (N, F), node_positions (N, 2), scaler
    Features: [temp, precip, temp_vol, temp_mom, precip_vol, precip_mom, gdp]
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
    """Build raw feature matrices for all timesteps. Returns list of (N,F) arrays."""
    from src.tail_risk.volatility import compute_volatility_series
    from src.tail_risk.momentum import compute_momentum_series

    tas, pr = data["tas"], data["pr"]
    T, _, _ = tas.shape

    temp_vol_series = compute_volatility_series(tas, window=5)
    temp_mom_series = compute_momentum_series(tas, window=3)
    precip_vol_series = compute_volatility_series(pr, window=5)
    precip_mom_series = compute_momentum_series(pr, window=3)

    features_list = []
    gdp_flat = data["gdp"].flatten()
    for t in range(T):
        feats = np.column_stack(
            [
                tas[t].flatten(),
                pr[t].flatten(),
                temp_vol_series[t].flatten(),
                temp_mom_series[t].flatten(),
                precip_vol_series[t].flatten(),
                precip_mom_series[t].flatten(),
                gdp_flat,
            ]
        )
        features_list.append(feats.astype(np.float32))

    return features_list


def build_temporal_features(data, scaler=None):
    """Build feature matrices for ALL timesteps. Returns list of (N,5) arrays.

    If `scaler` is provided (e.g., from `build_node_features`), features are transformed
    with it so temporal inference matches the training feature scale.
    """
    raw = build_temporal_features_raw(data)
    if scaler is None:
        return raw
    return [scaler.transform(f).astype(np.float32) for f in raw]
