"""Apply interventions and re-run GNN to get counterfactual risk."""
import numpy as np
import torch
from torch_geometric.data import Data


def apply_intervention(features, positions, intervention, lons, *, scaler=None):
    """
    Apply intervention deltas to feature matrix.

    Args:
        features: (N, F) numpy array — raw features from preprocess
        positions: (N, 2) numpy array — [lat, lon]
        intervention: dict from interventions.py
        lons: original lon values for coastal detection
        scaler: optional StandardScaler; if provided, returns scaled features

    Returns:
        modified_features: (N, F)
    """
    modified = features.copy()
    deltas = intervention['deltas']

    if deltas.get('coastal_only', False):
        threshold = deltas.get('coastal_lon_threshold', 120)
        mask = positions[:, 1] >= threshold
    else:
        mask = np.ones(len(features), dtype=bool)

    # Feature layout (see src.data.preprocess):
    # [temp, precip, temp_vol, temp_mom, precip_vol, precip_mom, gdp]

    # Temp reduction (feature index 0) — degrees C
    temp_delta = deltas.get('temp_reduction', 0)
    modified[mask, 0] -= temp_delta

    # Optional precip reduction (feature index 1) — mm/day (synthetic/demo)
    precip_delta = deltas.get("precip_reduction", 0)
    modified[mask, 1] -= precip_delta

    # Precip volatility reduction (feature index 4)
    vol_reduction = deltas.get('precip_volatility_reduction', 0)
    modified[mask, 4] *= (1 - vol_reduction)

    # Optional temp volatility reduction (feature index 2)
    temp_vol_reduction = deltas.get("temp_volatility_reduction", 0)
    modified[mask, 2] *= (1 - temp_vol_reduction)

    # Optional precip momentum reduction (feature index 5)
    precip_mom_reduction = deltas.get("precip_momentum_reduction", 0)
    modified[mask, 5] *= (1 - precip_mom_reduction)

    # GDP boost (feature index 6)
    gdp_factor = deltas.get('gdp_boost_factor', 1.0)
    modified[mask, 6] *= gdp_factor

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
