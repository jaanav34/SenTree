"""Preprocess climate data into node feature matrices."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_node_features(data, year_idx=-1):
    """
    Build feature matrix for a single timestep.
    Returns: features (N, 5), node_positions (N, 2), scaler
    Features: [temp, precip, volatility, momentum, gdp]
    """
    tas = data['tas']  # (T, nlat, nlon)
    pr = data['pr']

    # Flatten spatial dims
    temp = tas[year_idx].flatten()
    precip = pr[year_idx].flatten()
    gdp = data['gdp'].flatten()

    from src.tail_risk.volatility import compute_volatility
    from src.tail_risk.momentum import compute_momentum

    vol = compute_volatility(tas, window=5).flatten()
    mom = compute_momentum(tas, window=3).flatten()

    features = np.column_stack([temp, precip, vol, mom, gdp])

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    lats, lons = np.meshgrid(data['lats'], data['lons'], indexing='ij')
    positions = np.column_stack([lats.flatten(), lons.flatten()])

    return features.astype(np.float32), positions, scaler


def build_temporal_features(data):
    """Build feature matrices for ALL timesteps. Returns list of (N,5) arrays."""
    from src.tail_risk.volatility import compute_volatility_series
    from src.tail_risk.momentum import compute_momentum_series

    tas, pr = data['tas'], data['pr']
    T, nlat, nlon = tas.shape

    vol_series = compute_volatility_series(tas, window=5)
    mom_series = compute_momentum_series(tas, window=3)

    features_list = []
    for t in range(T):
        feats = np.column_stack([
            tas[t].flatten(),
            pr[t].flatten(),
            vol_series[t].flatten(),
            mom_series[t].flatten(),
            data['gdp'].flatten(),
        ])
        features_list.append(feats.astype(np.float32))

    return features_list
