"""Apply interventions and re-run GNN to get counterfactual risk."""
import numpy as np
import torch
from torch_geometric.data import Data


# Feature layout (src.data.preprocess, 11 features):
# [0] temp            [1] precip          [2] temp_vol
# [3] temp_mom        [4] precip_vol      [5] precip_mom
# [6] gdp             [7] pop_norm        [8] soil_moisture
# [9] coastal_factor  [10] tail_risk_score

_F = {
    'temp': 0, 'precip': 1, 'temp_vol': 2, 'temp_mom': 3,
    'precip_vol': 4, 'precip_mom': 5, 'gdp': 6, 'pop': 7,
    'soil': 8, 'coastal': 9, 'tail_risk': 10,
}


def apply_intervention(features, positions, intervention, lons, *, scaler=None, strength: float = 1.0):
    """
    Apply intervention deltas to raw feature matrix.

    Args:
        features: (N, 11) numpy array — raw features from preprocess
        positions: (N, 2) numpy array — [lat, lon]
        intervention: dict from interventions.py
        lons: original lon values for coastal detection
        scaler: optional StandardScaler; if provided, returns scaled features

    Returns:
        modified_features: (N, F) — scaled if scaler provided
    """
    if strength <= 0:
        raise ValueError("strength must be > 0")

    modified = features.copy()
    deltas = intervention['deltas']

    # Determine affected nodes
    if deltas.get('coastal_only', False):
        # Prefer a coastal-factor based mask (works globally) when available,
        # otherwise fall back to a longitude threshold (legacy / SE Asia demo).
        cf_thresh = deltas.get("coastal_factor_threshold", None)
        if cf_thresh is not None:
            mask = modified[:, _F["coastal"]] >= float(cf_thresh)
        else:
            threshold = deltas.get('coastal_lon_threshold', 120)
            mask = positions[:, 1] >= threshold
    else:
        mask = np.ones(len(features), dtype=bool)

    # Temperature reduction (degrees C)
    temp_delta = deltas.get('temp_reduction', 0) * strength
    modified[mask, _F['temp']] -= temp_delta

    # Precipitation reduction (mm/day)
    precip_delta = deltas.get('precip_reduction', 0) * strength
    modified[mask, _F['precip']] -= precip_delta

    # Temperature volatility reduction (fractional)
    temp_vol_reduction = deltas.get('temp_volatility_reduction', 0) * strength
    modified[mask, _F['temp_vol']] *= np.clip(1 - temp_vol_reduction, 0.0, 1.0)

    # Temperature momentum reduction (fractional)
    temp_mom_reduction = deltas.get('temp_momentum_reduction', 0) * strength
    modified[mask, _F['temp_mom']] *= np.clip(1 - temp_mom_reduction, 0.0, 1.0)

    # Precipitation volatility reduction (fractional)
    precip_vol_reduction = deltas.get('precip_volatility_reduction', 0) * strength
    modified[mask, _F['precip_vol']] *= np.clip(1 - precip_vol_reduction, 0.0, 1.0)

    # Precipitation momentum reduction (fractional)
    precip_mom_reduction = deltas.get('precip_momentum_reduction', 0) * strength
    modified[mask, _F['precip_mom']] *= np.clip(1 - precip_mom_reduction, 0.0, 1.0)

    # GDP boost (multiplicative)
    gdp_factor = 1.0 + (deltas.get('gdp_boost_factor', 1.0) - 1.0) * strength
    modified[mask, _F['gdp']] *= gdp_factor

    # Soil moisture improvement (additive)
    soil_boost = deltas.get('soil_moisture_boost', 0) * strength
    modified[mask, _F['soil']] = np.clip(modified[mask, _F['soil']] + soil_boost, 0, 1)

    # Tail-risk score reduction (the intervention directly lowers local risk)
    tail_risk_reduction = deltas.get('tail_risk_reduction', 0) * strength
    modified[mask, _F['tail_risk']] *= np.clip(1 - tail_risk_reduction, 0.0, 1.0)

    if scaler is not None:
        return scaler.transform(modified).astype(np.float32)

    return modified.astype(np.float32)


def run_all_simulations(
    model, base_data, features_raw, positions, interventions_dict, lons, *, scaler=None
):
    """Run all interventions, return results dict."""
    from src.model.gnn import predict

    baseline_risk = predict(model, base_data)
    results = {}

    for key, intervention in interventions_dict.items():
        mod_features = apply_intervention(
            features_raw, positions, intervention, lons, scaler=scaler
        )
        mod_data = Data(
            x=torch.tensor(mod_features, dtype=torch.float32),
            edge_index=base_data.edge_index,
            pos=base_data.pos,
            num_nodes=base_data.num_nodes,
        )
        int_risk = predict(model, mod_data)

        results[key] = {
            'name': intervention['name'],
            'baseline_risk': baseline_risk,
            'intervention_risk': int_risk,
            'risk_reduction': baseline_risk - int_risk,
            'cost': intervention['cost_usd'],
        }

    return baseline_risk, results
