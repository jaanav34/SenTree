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
from src.tail_risk.engine import compute_tail_risk, compute_tail_risk_series, get_tail_risk_nodes
from src.graph.build_graph import build_climate_graph
from src.model.gnn import ClimateRiskGNN, train_gnn, predict
from src.simulation.interventions import INTERVENTIONS
from src.simulation.run_simulations import run_all_simulations, apply_intervention
from src.simulation.roi import compute_roi
from src.rendering.render_video import (
    render_risk_video, render_comparison_video, render_tail_risk_video
)

print("=" * 60)
print("SenTree Pipeline")
print("=" * 60)

# 1. Load data
print("\n[1/7] Loading climate data...")
data = load_climate_data()
print(f"  Shape: {data['tas'].shape} — {len(data['years'])} years, "
      f"{data['tas'].shape[1]}x{data['tas'].shape[2]} grid")

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
model = train_gnn(model, graph_data, tail_scores_flat, epochs=50)

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

os.makedirs('outputs/roi', exist_ok=True)
with open('outputs/roi/roi_results.json', 'w') as f:
    json.dump(roi_results, f, indent=2, default=str)

# 7. Render videos
print("\n[7/7] Rendering videos...")
nlat, nlon = data['tas'].shape[1], data['tas'].shape[2]
T = data['tas'].shape[0]

temporal_features_raw = build_temporal_features_raw(data)
temporal_features = build_temporal_features(data, scaler=scaler)
baseline_risk_series = []
intervention_risk_series = {key: [] for key in INTERVENTIONS}

import torch
from torch_geometric.data import Data as PyGData

for t in range(T):
    feats = temporal_features[t]
    feats_raw = temporal_features_raw[t]
    temp_data = PyGData(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=graph_data.edge_index,
        pos=graph_data.pos,
        num_nodes=graph_data.num_nodes,
    )
    b_risk = predict(model, temp_data)
    baseline_risk_series.append(b_risk.reshape(nlat, nlon))

    for key, interv in INTERVENTIONS.items():
        mod_feats = apply_intervention(feats_raw, positions, interv, lons, scaler=scaler)
        mod_data = PyGData(
            x=torch.tensor(mod_feats, dtype=torch.float32),
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

years = [int(y) for y in range(2015, 2015 + T)]
risk_timeseries = {"years": years, "baseline": _series_stats(baseline_risk_series)}
for key in INTERVENTIONS:
    risk_timeseries[key] = _series_stats(intervention_risk_series[key])

os.makedirs("outputs/roi", exist_ok=True)
with open("outputs/roi/risk_timeseries.json", "w") as f:
    json.dump(risk_timeseries, f, indent=2)

# Tail-risk flags per timestep (using full Gurjar & Camp engine)
print("  Computing per-timestep tail-risk flags...")
_scores_series, flags_series, _regime_series = compute_tail_risk_series(data)

# Render all videos
render_risk_video(baseline_risk_series, data['lats'], data['lons'],
                  'outputs/videos/baseline_risk.mp4', title='Baseline Climate Risk')

render_tail_risk_video(baseline_risk_series, flags_series, data['lats'], data['lons'],
                       'outputs/videos/tail_risk_escalation.mp4')

for key in INTERVENTIONS:
    name = INTERVENTIONS[key]['name']
    render_comparison_video(
        baseline_risk_series, intervention_risk_series[key],
        data['lats'], data['lons'],
        f'outputs/videos/comparison_{key}.mp4',
        intervention_name=name
    )
    render_risk_video(
        intervention_risk_series[key], data['lats'], data['lons'],
        f'outputs/videos/{key}_risk.mp4',
        title=f'Risk with {name}'
    )

print("\n" + "=" * 60)
print("Pipeline complete!")
print(f"Videos: outputs/videos/")
print(f"ROI data: outputs/roi/roi_results.json")
print("=" * 60)
