"""End-to-end pipeline: data -> tail risk -> graph -> GNN -> simulations -> videos."""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

import numpy as np

from src.data.load_isimip import load_climate_data
from src.data.preprocess import (
    build_node_features,
    build_node_features_raw,
    build_temporal_features,
    build_temporal_features_raw,
)
from src.data.koppen_geiger import classify_grid
from src.tail_risk.engine import compute_tail_risk, compute_tail_risk_series, get_tail_risk_nodes
from src.graph.build_graph import build_climate_graph
from src.model.gnn import ClimateRiskGNN, train_gnn, predict
from src.simulation.interventions import INTERVENTIONS
from src.simulation.run_simulations import run_all_simulations, apply_intervention
from src.simulation.roi import compute_roi
from src.rendering.render_video import (
    render_risk_video, render_comparison_video, render_tail_risk_video, render_tail_risk_map, render_kg_video
)

print("=" * 60)
print("SenTree Pipeline")
print("=" * 60)

# 1. Load data
print("\n[1/7] Loading climate data...")
region = os.environ.get("SENTREE_REGION", "se_asia")
coarsen = int(os.environ.get("SENTREE_COARSEN", "1"))
intervention_strength = float(os.environ.get("SENTREE_INTERVENTION_STRENGTH", "1.0"))
data = load_climate_data(region=region, coarsen=coarsen)
years = data['years']
T = len(years)
nlat, nlon = data['tas'].shape[1], data['tas'].shape[2]
print(f"  Shape: {data['tas'].shape} — {T} years, {nlat}x{nlon} grid")
print(f"  Years: {years[0]} to {years[-1]}")
print(f"  Region: {region} | Coarsen: {coarsen}x | Intervention strength: {intervention_strength:g}x")

# Precompute Köppen-Geiger codes for climate-relative stabilization
print("  Precomputing Köppen-Geiger climate classification...")
kg_cache_path = f"data/processed/kg_codes_{region}_c{int(coarsen)}.npz"
try:
    kg_cached = np.load(kg_cache_path, allow_pickle=False)
    kg_series = kg_cached["kg_codes"].astype(np.int32)
    if kg_series.shape[:1] != (len(years),):
        raise ValueError(f"cached kg_codes has wrong time dim: {kg_series.shape}")
    data["kg_codes"] = kg_series
    print(f"  Loaded cached KG codes from {kg_cache_path}")
except Exception:
    kg_series = classify_grid(data['tas_monthly'], data['pr_monthly'])
    data["kg_codes"] = kg_series.astype(np.int32)
    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(kg_cache_path, kg_codes=data["kg_codes"], years=np.asarray(years, dtype=np.int32))
    print(f"  Saved cached KG codes to {kg_cache_path}")

# 2. Compute tail risk (Gurjar & Camp 2026 + Hawkes process)
print("\n[2/7] Computing tail-risk scores...")
scores, flags, threshold, regime, components = compute_tail_risk(data)
print(f"  Threshold (95th pct): {threshold:.4f}")
print(f"  Flagged nodes: {flags.sum()} / {flags.size}")
n_baseline = int((regime == 0).sum())
n_buildup = int((regime == 1).sum())
n_surge = int((regime == 2).sum())
print(f"  Regimes — Baseline: {n_baseline}, Buildup: {n_buildup}, Surge: {n_surge}")

# 3. Build features + graph
print("\n[3/7] Building graph...")
features, positions, scaler = build_node_features(data, year_idx=-1)
features_raw, _positions_raw = build_node_features_raw(data, year_idx=-1)
n_features = features.shape[1]
print(f"  Feature dimensions: {n_features}")
graph_data = build_climate_graph(features, positions, k=8)
print(f"  Nodes: {graph_data.num_nodes}, Edges: {graph_data.edge_index.shape[1]}")

# 4. Train GNN (upgraded GAT+GCN hybrid)
print("\n[4/7] Training GNN...")
model = ClimateRiskGNN(in_channels=n_features, hidden_channels=64)
tail_scores_flat, _, _ = get_tail_risk_nodes(data)
model, training_history = train_gnn(
    model, graph_data, tail_scores_flat, epochs=50, return_history=True
)

os.makedirs('outputs/roi', exist_ok=True)
np.savez_compressed(
    'outputs/roi/gnn_training_history.npz',
    positions=training_history['positions'],
    edge_index=training_history['edge_index_sample'],
    target=training_history['target'],
    predictions=training_history['predictions'],
    loss=training_history['loss'],
    learning_rate=training_history['learning_rate'],
)
print("  Saved training history: outputs/roi/gnn_training_history.npz")

# 5. Run simulations
print("\n[5/7] Running simulations...")
lons = data['lons']
baseline_risk, sim_results = run_all_simulations(
    model, graph_data, features_raw, positions, INTERVENTIONS, lons, scaler=scaler
)

# 6. Compute ROI (with Ito 2020 ensemble uncertainty)
print("\n[6/7] Computing ROI...")
roi_results = {}
for key, result in sim_results.items():
    roi = compute_roi(
        result['baseline_risk'], result['intervention_risk'],
        result['cost'], data['gdp'].flatten(), data['pop'].flatten(),
        precip_data=data['pr']
    )
    roi_results[key] = {
        'name': result['name'],
        **roi,
        'tail_risk_nodes_neutralized': int((
            (result['baseline_risk'] > np.percentile(result['baseline_risk'], 95)) &
            (result['intervention_risk'] <= np.percentile(result['baseline_risk'], 95))
        ).sum())
    }
    print(f"  {result['name']}:")
    print(f"    ROI = {roi['roi']:.2f}x (range: {roi['roi_lower']:.2f} - {roi['roi_upper']:.2f})")
    print(f"    Loss avoided: ${roi['total_loss_avoided']:,.0f}")
    print(f"    Risk reduction: {roi['mean_risk_reduction']:.4f} (mean), {roi['max_risk_reduction']:.4f} (max)")
    print(f"    Uncertainty: U_precip={roi['u_precip']:.3f}, U_model={roi['u_model']:.3f}, U_scenario={roi['u_scenario']:.3f}")
    print(f"    FRA (Ito 2020): {roi['fra']:.3f}")

with open('outputs/roi/roi_results.json', 'w') as f:
    json.dump(roi_results, f, indent=2, default=str)

# 7. Render videos
print("\n[7/7] Rendering videos...")

temporal_features_raw = build_temporal_features_raw(data)
temporal_features, scaler = build_temporal_features(data)
baseline_risk_series = []
intervention_risk_series = {key: [] for key in INTERVENTIONS}

import torch
from torch_geometric.data import Data as PyGData

for t in range(T):
    feats = temporal_features[t]
    feats_raw = temporal_features_raw[t]
    temp_data = PyGData(
        x=torch.from_numpy(np.asarray(feats, dtype=np.float32)),
        edge_index=graph_data.edge_index,
        pos=graph_data.pos,
        num_nodes=graph_data.num_nodes,
    )

    b_risk = predict(model, temp_data)
    baseline_risk_series.append(b_risk.reshape(nlat, nlon))

    for key, interv in INTERVENTIONS.items():
        mod_feats = apply_intervention(
            feats_raw, positions, interv, lons, scaler=scaler, strength=intervention_strength
        )
        mod_data = PyGData(
            x=torch.from_numpy(np.asarray(mod_feats, dtype=np.float32)),
            edge_index=graph_data.edge_index,
            pos=graph_data.pos,
            num_nodes=graph_data.num_nodes,
        )
        i_risk = predict(model, mod_data)
        intervention_risk_series[key].append(i_risk.reshape(nlat, nlon))

# Save quantitative time-series metrics
def _series_stats(series_2d):
    return {
        "mean": [float(x.mean()) for x in series_2d],
        "p95": [float(np.percentile(x, 95)) for x in series_2d],
        "max": [float(x.max()) for x in series_2d],
    }

risk_timeseries = {
    "years": [int(y) for y in years],
    "baseline": _series_stats(baseline_risk_series),
}
for key in INTERVENTIONS:
    risk_timeseries[key] = _series_stats(intervention_risk_series[key])

os.makedirs("outputs/roi", exist_ok=True)
with open("outputs/roi/risk_timeseries.json", "w") as f:
    json.dump(risk_timeseries, f, indent=2)

# Tail-risk flags per timestep (using full Gurjar & Camp engine)
print("  Computing per-timestep tail-risk flags...")
_scores_series, flags_series, _regime_series = compute_tail_risk_series(data)

# Render all videos (pass year_labels for correct annotation)
year_labels = years

render_risk_video(baseline_risk_series, data['lats'], data['lons'],
                  'outputs/videos/baseline_risk.mp4', title='Baseline Climate Risk',
                  year_labels=year_labels)

render_tail_risk_video(baseline_risk_series, flags_series, data['lats'], data['lons'],
                       'outputs/videos/tail_risk_escalation.mp4',
                       year_labels=year_labels)

render_kg_video(kg_series, data['lats'], data['lons'],
                'outputs/videos/climate_classification_shift.mp4',
                year_labels=year_labels)

print("  Generating Strategic Resilience Opportunity Map...")
# Logic: Map the DELTA (Baseline - Intervention) to show where value is created
# We aggregate reduction across all interventions to find the 'Value hotspots'
total_reduction_map = np.zeros((nlat, nlon))

for key in sim_results:
    # Get the 2D reduction potential for this intervention
    reduction = (sim_results[key]['baseline_risk'] - sim_results[key]['intervention_risk']).reshape(nlat, nlon)
    total_reduction_map += reduction

render_tail_risk_map(
    total_reduction_map,
    flags_series[-1],
    data['lats'],
    data['lons'],
    'outputs/tail_risk_map.png',
    title='Strategic Resilience Opportunity & ROI Target Map',
    label='Total Avoided Damage Potential (Risk Reduction)'
)

# Save underlying grids for interactive dashboard overlays
os.makedirs("outputs/roi", exist_ok=True)
np.savez_compressed(
    "outputs/roi/opportunity_map.npz",
    total_reduction_map=total_reduction_map.astype(np.float32),
    tail_flags=flags_series[-1].astype(np.uint8),
    lats=np.asarray(data["lats"], dtype=np.float64),
    lons=np.asarray(data["lons"], dtype=np.float64),
    years=np.asarray(years, dtype=np.int32),
)

# PRINT COORDINATES TO CLI FOR IMMEDIATE ACTION
flagged_lats = data['lats'].repeat(nlon)[flags_series[-1].flatten()]
flagged_lons = np.tile(data['lons'], nlat)[flags_series[-1].flatten()]
print(f"\n  HIGH-ROI TARGET LOCATIONS (Top 5 Priority Nodes):")
for i in range(min(5, len(flagged_lats))):
    print(f"    - Coordinate: {flagged_lons[i]:.2f}E, {flagged_lats[i]:.2f}N")

for key in INTERVENTIONS:
    name = INTERVENTIONS[key]['name']
    render_comparison_video(
        baseline_risk_series, intervention_risk_series[key],
        data['lats'], data['lons'],
        f'outputs/videos/comparison_{key}.mp4',
        intervention_name=name,
        year_labels=year_labels,
    )

print("\n" + "=" * 60)
print("Pipeline complete!")
print(f"Videos: outputs/videos/")
print(f"ROI data: outputs/roi/roi_results.json")
print("=" * 60)
