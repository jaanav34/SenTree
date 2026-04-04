"""End-to-end pipeline: data -> tail risk -> graph -> GNN -> simulations -> videos."""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_isimip import load_climate_data
from src.data.preprocess import build_node_features, build_temporal_features
from src.tail_risk.engine import compute_tail_risk, get_tail_risk_nodes
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

# 2. Compute tail risk
print("\n[2/7] Computing tail-risk scores...")
scores, flags, threshold = compute_tail_risk(data)
print(f"  Threshold (95th pct): {threshold:.4f}")
print(f"  Flagged nodes: {flags.sum()} / {flags.size}")

# 3. Build features + graph
print("\n[3/7] Building graph...")
features, positions, scaler = build_node_features(data, year_idx=-1)
graph_data = build_climate_graph(features, positions, k=8)
print(f"  Nodes: {graph_data.num_nodes}, Edges: {graph_data.edge_index.shape[1]}")

# 4. Train GNN
print("\n[4/7] Training GNN...")
model = ClimateRiskGNN(in_channels=features.shape[1])
tail_scores_flat, _, _ = get_tail_risk_nodes(data)
model = train_gnn(model, graph_data, tail_scores_flat, epochs=50)

# 5. Run simulations
print("\n[5/7] Running simulations...")
lons = data['lons']
baseline_risk, sim_results = run_all_simulations(
    model, graph_data, features, positions, INTERVENTIONS, lons
)

# 6. Compute ROI
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
    print(f"  {result['name']}: ROI = {roi['roi']:.2f}x "
          f"(range: {roi['roi_lower']:.2f} - {roi['roi_upper']:.2f})")

os.makedirs('outputs/roi', exist_ok=True)
with open('outputs/roi/roi_results.json', 'w') as f:
    json.dump(roi_results, f, indent=2, default=str)

# 7. Render videos
print("\n[7/7] Rendering videos...")
nlat, nlon = data['tas'].shape[1], data['tas'].shape[2]
T = data['tas'].shape[0]

temporal_features = build_temporal_features(data)
baseline_risk_series = []
intervention_risk_series = {key: [] for key in INTERVENTIONS}

import torch
from torch_geometric.data import Data as PyGData

for t in range(T):
    feats = temporal_features[t]
    temp_data = PyGData(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=graph_data.edge_index,
        pos=graph_data.pos,
        num_nodes=graph_data.num_nodes,
    )
    b_risk = predict(model, temp_data)
    baseline_risk_series.append(b_risk.reshape(nlat, nlon))

    for key, interv in INTERVENTIONS.items():
        mod_feats = apply_intervention(feats, positions, interv, lons)
        mod_data = PyGData(
            x=torch.tensor(mod_feats, dtype=torch.float32),
            edge_index=graph_data.edge_index,
            pos=graph_data.pos,
            num_nodes=graph_data.num_nodes,
        )
        i_risk = predict(model, mod_data)
        intervention_risk_series[key].append(i_risk.reshape(nlat, nlon))

# Tail-risk flags per timestep
from src.tail_risk.volatility import compute_volatility_series
from src.tail_risk.momentum import compute_momentum_series

vol_s = compute_volatility_series(data['tas'], 5)
mom_s = compute_momentum_series(data['tas'], 3)

flags_series = []
for t in range(T):
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)
    s = 0.6 * norm(vol_s[t]) + 0.4 * norm(mom_s[t])
    flags_series.append(s >= np.percentile(s, 95))

# Render
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
