"""Apply interventions and re-run GNN to get counterfactual risk."""
import numpy as np
import torch
from torch_geometric.data import Data


def apply_intervention(features, positions, intervention, lons):
    """
    Apply intervention deltas to feature matrix.

    Args:
        features: (N, 5) numpy array — [temp, precip, volatility, momentum, gdp]
        positions: (N, 2) numpy array — [lat, lon]
        intervention: dict from interventions.py
        lons: original lon values for coastal detection

    Returns:
        modified_features: (N, 5)
    """
    modified = features.copy()
    deltas = intervention['deltas']

    if deltas.get('coastal_only', False):
        threshold = deltas.get('coastal_lon_threshold', 120)
        mask = positions[:, 1] >= threshold
    else:
        mask = np.ones(len(features), dtype=bool)

    # Temp reduction (feature index 0)
    temp_delta = deltas.get('temp_reduction', 0)
    modified[mask, 0] -= temp_delta * 0.5

    # Volatility reduction (feature index 2)
    vol_reduction = deltas.get('precip_volatility_reduction', 0)
    modified[mask, 2] *= (1 - vol_reduction)

    # GDP boost (feature index 4)
    gdp_factor = deltas.get('gdp_boost_factor', 1.0)
    modified[mask, 4] *= gdp_factor

    return modified


def run_all_simulations(model, base_data, features, positions, interventions_dict, lons):
    """Run all interventions, return results dict."""
    from src.model.gnn import predict

    baseline_risk = predict(model, base_data)
    results = {}

    for key, intervention in interventions_dict.items():
        mod_features = apply_intervention(features, positions, intervention, lons)
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
