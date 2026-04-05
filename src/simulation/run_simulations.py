"""Apply interventions and re-run GNN to get counterfactual risk."""
import numpy as np
import torch
from torch_geometric.data import Data

from src.data.koppen_geiger import KG_LABELS

# Feature layout (src.data.preprocess, 43 features):
# [0] temp            [1] precip          [2] temp_vol
# [3] temp_mom        [4] precip_vol      [5] precip_mom
# [6] gdp             [7] pop_norm        [8] soil_moisture
# [9] coastal_factor  [10] tail_risk_score [11..42] KG one-hot

_F = {
    'temp': 0, 'precip': 1, 'temp_vol': 2, 'temp_mom': 3,
    'precip_vol': 4, 'precip_mom': 5, 'gdp': 6, 'pop': 7,
    'soil': 8, 'coastal': 9, 'tail_risk': 10,
}
_KG_START = 11


def _kg_mask(features: np.ndarray, deltas: dict) -> np.ndarray:
    """Return a KG compatibility mask derived from the one-hot climate columns."""
    if features.shape[1] <= _KG_START:
        return np.ones(features.shape[0], dtype=bool)

    kg_onehot = np.asarray(features[:, _KG_START:], dtype=np.float32)
    if kg_onehot.shape[1] == 0:
        return np.ones(features.shape[0], dtype=bool)

    kg_idx = np.argmax(kg_onehot, axis=1)
    kg_codes = np.array([KG_LABELS.get(int(idx), "Unknown") for idx in kg_idx], dtype=object)

    allow_codes = set(deltas.get("kg_allow_codes", []))
    allow_prefixes = tuple(deltas.get("kg_allow_prefixes", []))
    block_codes = set(deltas.get("kg_block_codes", []))
    block_prefixes = tuple(deltas.get("kg_block_prefixes", []))

    mask = np.ones(features.shape[0], dtype=bool)
    if allow_codes or allow_prefixes:
        mask &= np.array(
            [
                bool((code in allow_codes) or (allow_prefixes and str(code).startswith(allow_prefixes)))
                for code in kg_codes
            ],
            dtype=bool,
        )
    if block_codes or block_prefixes:
        mask &= np.array(
            [
                bool((code not in block_codes) and not (block_prefixes and str(code).startswith(block_prefixes)))
                for code in kg_codes
            ],
            dtype=bool,
        )
    return mask


def _condition_mask(features: np.ndarray, deltas: dict) -> np.ndarray:
    """Return non-KG applicability constraints for an intervention."""
    mask = np.ones(features.shape[0], dtype=bool)

    min_coastal = deltas.get("min_coastal_factor")
    if min_coastal is not None:
        mask &= features[:, _F["coastal"]] >= float(min_coastal)

    max_coastal = deltas.get("max_coastal_factor")
    if max_coastal is not None:
        mask &= features[:, _F["coastal"]] <= float(max_coastal)

    min_soil = deltas.get("min_soil_moisture")
    if min_soil is not None:
        mask &= features[:, _F["soil"]] >= float(min_soil)

    max_soil = deltas.get("max_soil_moisture")
    if max_soil is not None:
        mask &= features[:, _F["soil"]] <= float(max_soil)

    min_temp = deltas.get("min_temp")
    if min_temp is not None:
        mask &= features[:, _F["temp"]] >= float(min_temp)

    max_temp = deltas.get("max_temp")
    if max_temp is not None:
        mask &= features[:, _F["temp"]] <= float(max_temp)

    min_precip = deltas.get("min_precip")
    if min_precip is not None:
        mask &= features[:, _F["precip"]] >= float(min_precip)

    max_precip = deltas.get("max_precip")
    if max_precip is not None:
        mask &= features[:, _F["precip"]] <= float(max_precip)

    return mask


def get_intervention_mask(features, positions, intervention, lons):
    """Compute the nodes eligible for a given intervention."""
    deltas = intervention['deltas']

    if deltas.get('coastal_only', False):
        cf_thresh = deltas.get("coastal_factor_threshold", None)
        if cf_thresh is not None:
            mask = features[:, _F["coastal"]] >= float(cf_thresh)
        else:
            threshold = deltas.get('coastal_lon_threshold', 120)
            mask = positions[:, 1] >= threshold
    else:
        mask = np.ones(len(features), dtype=bool)

    mask &= _condition_mask(features, deltas)
    mask &= _kg_mask(features, deltas)
    return mask


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
    mask = get_intervention_mask(modified, positions, intervention, lons)

    # Temperature reduction (degrees C)
    temp_delta = deltas.get('temp_reduction', 0) * strength
    modified[mask, _F['temp']] -= temp_delta

    # Precipitation reduction (mm/day)
    precip_delta = deltas.get('precip_reduction', 0) * strength
    modified[mask, _F['precip']] = np.maximum(modified[mask, _F['precip']] - precip_delta, 0.0)

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
    model, base_data, features_raw, positions, interventions_dict, lons, *, scaler=None, strength: float = 1.0
):
    """Run all interventions, return results dict."""
    from src.model.gnn import predict

    baseline_risk = predict(model, base_data)
    results = {}

    for key, intervention in interventions_dict.items():
        eligible_mask = get_intervention_mask(features_raw, positions, intervention, lons)
        mod_features = apply_intervention(
            features_raw, positions, intervention, lons, scaler=scaler, strength=strength
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
            'eligible_nodes': int(eligible_mask.sum()),
            'eligible_share': float(np.mean(eligible_mask)),
        }

    return baseline_risk, results
