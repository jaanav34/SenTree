"""
Climate Risk GNN — KG-Enhanced Architecture.

Upgrades from baseline:
  1. KGAwareEdgeAttention: Injects a KG-class-similarity bias into GAT
     attention scores. Nodes sharing the same climate regime attend more
     strongly to each other, enabling cross-continental risk transfer
     (e.g., Amazon Af nodes inform Congo Basin Af nodes).

  2. KGGatedReadout: Final risk score is modulated by a per-KG-class
     vulnerability embedding. A 2°C rise in BWh (Hot Desert) triggers
     a different damage response than the same rise in Cfb (Temperate
     Oceanic). This replaces the single linear readout head.

  3. KG Regime Contrastive Loss: Training loss includes a term that
     penalizes the model for mapping nodes with different KG classes +
     same temperature anomaly to the same risk score. Forces the model
     to respect climate-zone-specific risk dynamics.

References:
  - Velickovic et al. (2018) "Graph Attention Networks"
  - Gurjar & Camp (2026) for the feature engineering rationale
  - Beck et al. (2018) "Present and future Köppen-Geiger climate
    classification maps at 1-km resolution"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
import numpy as np


def _get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# KG vulnerability priors (Beck et al. 2018 + IPCC AR6 damage literature)
# Maps KG class index (0-31) to baseline vulnerability scalar in [0, 1].
# Higher = more sensitive to the same temperature/precip anomaly.
# ---------------------------------------------------------------------------
_KG_VULNERABILITY = {
    0:  0.75,   # Af  — Tropical Rainforest
    1:  0.72,   # Am  — Tropical Monsoon
    2:  0.78,   # Aw  — Tropical Savanna
    3:  0.90,   # BWh — Hot Desert (most vulnerable globally)
    4:  0.85,   # BWk — Cold Desert
    5:  0.82,   # BSh — Hot Semi-Arid
    6:  0.78,   # BSk — Cold Semi-Arid
    7:  0.45,   # Csa — Hot-summer Mediterranean
    8:  0.42,   # Csb — Warm-summer Mediterranean
    9:  0.38,   # Csc — Cold-summer Mediterranean
    10: 0.50,   # Cwa — Monsoon-influenced humid subtropical
    11: 0.48,   # Cwb — Subtropical highland
    12: 0.44,   # Cwc — Cold subtropical highland
    13: 0.40,   # Cfa — Humid subtropical
    14: 0.38,   # Cfb — Oceanic (most resilient temperate)
    15: 0.35,   # Cfc — Subpolar oceanic
    16: 0.55,   # Dsa
    17: 0.52,   # Dsb
    18: 0.50,   # Dsc
    19: 0.48,   # Dsd
    20: 0.58,   # Dwa
    21: 0.55,   # Dwb
    22: 0.52,   # Dwc
    23: 0.50,   # Dwd
    24: 0.53,   # Dfa
    25: 0.50,   # Dfb
    26: 0.48,   # Dfc
    27: 0.45,   # Dfd
    28: 0.65,   # ET — Tundra
    29: 0.60,   # EF — Ice Cap
    30: 0.50,
    31: 0.50,
}


def _build_kg_vulnerability_tensor(n_classes: int = 32) -> torch.Tensor:
    v = torch.zeros(n_classes)
    for i in range(n_classes):
        v[i] = _KG_VULNERABILITY.get(i, 0.50)
    return v


# ---------------------------------------------------------------------------
# KG Edge Bias
# ---------------------------------------------------------------------------

class KGEdgeBias(nn.Module):
    """
    Learnable attention bias for edges connecting same-KG-class nodes.
    Added to GAT logits so same-regime nodes attend more strongly.
    Cost: O(E) — negligible.
    """
    def __init__(self, init_same_bias: float = 0.5):
        super().__init__()
        self.same_class_bias = nn.Parameter(torch.tensor(init_same_bias))

    def forward(self, kg_onehot: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        kg_class = kg_onehot.argmax(dim=-1)
        same_class = (kg_class[src] == kg_class[dst]).float()
        return same_class * self.same_class_bias


# ---------------------------------------------------------------------------
# Climate-Conditioned GAT with KG Edge Bias
# ---------------------------------------------------------------------------

class ClimateConditionedGAT(nn.Module):
    """
    GAT layer conditioned on Köppen-Geiger prior with KG edge bias.
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.15):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads,
                           dropout=dropout, concat=False)
        self.kg_prior = nn.Linear(32, out_channels)
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
        self.kg_edge_bias = KGEdgeBias(init_same_bias=0.5)

    def forward(self, x, edge_index, kg_onehot):
        edge_bias = self.kg_edge_bias(kg_onehot, edge_index)
        x_gat = self.gat(x, edge_index, edge_attr=edge_bias.unsqueeze(-1))
        prior = self.kg_prior(kg_onehot)
        gate = torch.sigmoid(self.gate)
        return (1 - gate) * x_gat + gate * prior


# ---------------------------------------------------------------------------
# KG-Gated Readout
# ---------------------------------------------------------------------------

class KGGatedReadout(nn.Module):
    """
    Risk readout head modulated by per-KG-class vulnerability embeddings.

    risk = sigmoid(mlp(hidden) + vuln_scale * kg_vulnerability[class])

    Forces desert nodes (BWh, vuln=0.90) to produce higher risk than
    oceanic nodes (Cfb, vuln=0.38) given identical hidden states.
    """
    def __init__(self, hidden_channels: int, n_kg_classes: int = 32,
                 dropout: float = 0.15):
        super().__init__()
        vuln_prior = _build_kg_vulnerability_tensor(n_kg_classes)
        self.kg_vulnerability = nn.Parameter(vuln_prior.unsqueeze(-1))  # (32, 1)
        self.vuln_scale = nn.Parameter(torch.tensor(0.3))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor, kg_onehot: torch.Tensor) -> torch.Tensor:
        raw_logit = self.mlp(x)                              # (N, 1)
        node_vuln = kg_onehot @ self.kg_vulnerability        # (N, 1)
        vuln_scale = torch.sigmoid(self.vuln_scale)
        adjusted = raw_logit + vuln_scale * node_vuln
        return torch.sigmoid(adjusted).squeeze(-1)           # (N,)


# ---------------------------------------------------------------------------
# Main GNN
# ---------------------------------------------------------------------------

class ClimateRiskGNN(nn.Module):
    """
    Climate-Conditioned GAT + GCN hybrid with full KG integration.

    Layer 1: ClimateConditionedGAT  — KG prior + same-class edge bias
    Layer 2: GCNConv                — broad spatial propagation
    Layer 3: GATConv                — refined attention
    Layer 4: GCNConv                — final aggregation
    Readout: KGGatedReadout         — vulnerability-modulated sigmoid

    Input: 43-dim [11 continuous + 32 KG one-hot]
    """

    def __init__(self, in_channels: int = 43, hidden_channels: int = 64,
                 out_channels: int = 1, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.dropout_rate = dropout

        self.conditioned_gat1 = ClimateConditionedGAT(
            in_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.bn1 = BatchNorm(hidden_channels)

        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.gat3 = GATConv(hidden_channels, hidden_channels, heads=2,
                            dropout=dropout, concat=False)
        self.bn3 = BatchNorm(hidden_channels)

        self.gcn4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = BatchNorm(hidden_channels)

        self.skip_proj = nn.Linear(in_channels, hidden_channels)
        self.readout = KGGatedReadout(hidden_channels, n_kg_classes=32,
                                      dropout=dropout)

    def forward(self, data, mc_dropout: bool = False):
        x, edge_index = data.x, data.edge_index
        kg_onehot = x[:, -32:]

        if mc_dropout:
            self.train()

        skip = self.skip_proj(x)
        x = self.conditioned_gat1(x, edge_index, kg_onehot)
        x = self.bn1(x)
        x = F.elu(x)
        x = x + skip

        skip = x
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x + skip

        skip = x
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = x + skip

        skip = x
        x = self.gcn4(x, edge_index)
        x = self.bn4(x)
        x = F.elu(x)
        x = x + skip

        return self.readout(x, kg_onehot)


# ---------------------------------------------------------------------------
# KG Regime Contrastive Loss
# ---------------------------------------------------------------------------

def kg_regime_loss(pred: torch.Tensor, kg_onehot: torch.Tensor,
                   features: torch.Tensor, temp_idx: int = 0,
                   weight: float = 0.05) -> torch.Tensor:
    """
    Penalizes identical risk predictions for nodes with different KG
    classes but similar temperature anomalies.

    A 2°C anomaly in BWh (desert, vuln=0.90) must produce higher risk
    than in Cfb (oceanic, vuln=0.38). Without this term the model can
    ignore KG entirely and still minimize Huber loss.

    Samples up to 512 random pairs per step — O(N), not O(N²).
    """
    if weight == 0:
        return torch.tensor(0.0, device=pred.device)

    N = pred.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=pred.device)

    kg_class = kg_onehot.argmax(dim=-1)
    temp = features[:, temp_idx]

    n_pairs = min(512, N * (N - 1) // 2)
    idx_i = torch.randint(0, N, (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, N, (n_pairs,), device=pred.device)

    temp_tol = temp.std() * 0.5
    diff_kg   = (kg_class[idx_i] != kg_class[idx_j])
    similar_t = (temp[idx_i] - temp[idx_j]).abs() < temp_tol
    valid = diff_kg & similar_t

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    i_v, j_v = idx_i[valid], idx_j[valid]

    vuln = torch.tensor(
        [_KG_VULNERABILITY.get(int(c), 0.5) for c in kg_class.cpu().numpy()],
        device=pred.device, dtype=torch.float32,
    )
    margin = (vuln[i_v] - vuln[j_v]).abs().clamp(min=0.05)
    risk_diff = (pred[i_v] - pred[j_v]).abs()

    return weight * F.relu(margin - risk_diff).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gnn(model, data, target_scores, epochs=50, lr=0.005,
              weight_decay=1e-4, schedule=True, kg_loss_weight=0.05,
              return_history: bool = False):
    """
    AdamW + cosine annealing + label smoothing + KG regime contrastive loss.
    """
    device = _get_default_device()
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = None
    if schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

    target = torch.as_tensor(target_scores, dtype=torch.float32, device=device)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)
    target = target * 0.95 + 0.025

    loss_fn = nn.HuberLoss(delta=0.5)
    best_loss = float('inf')
    loss_history = []
    learning_rate_history = []
    prediction_history = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(data)

        loss = loss_fn(pred, target)

        if data.edge_index.shape[1] > 0:
            src, dst = data.edge_index[0], data.edge_index[1]
            loss = loss + 0.01 * torch.mean((pred[src] - pred[dst]) ** 2)

        kg_onehot = data.x[:, -32:]
        loss = loss + kg_regime_loss(pred, kg_onehot, data.x,
                                     weight=kg_loss_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()

        loss_history.append(loss.item())
        learning_rate_history.append(current_lr)

        if return_history:
            model.eval()
            with torch.no_grad():
                prediction_history.append(
                    model(data).detach().cpu().numpy().copy()
                )
            model.train()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, "
                  f"LR: {current_lr:.6f}")

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()

    if not return_history:
        return model

    positions = getattr(data, "pos", None)
    edge_index = getattr(data, "edge_index", None)

    history = {
        "positions": positions.detach().cpu().numpy().copy()
        if positions is not None else np.empty((0, 2), dtype=np.float32),
        "edge_index_sample": edge_index.detach().cpu().numpy().copy()
        if edge_index is not None else np.empty((2, 0), dtype=np.int64),
        "target": target.detach().cpu().numpy().copy(),
        "predictions": np.stack(prediction_history).astype(np.float32, copy=False)
        if prediction_history else np.empty((0, target.shape[0]), dtype=np.float32),
        "loss": np.asarray(loss_history, dtype=np.float32),
        "learning_rate": np.asarray(learning_rate_history, dtype=np.float32),
    }

    return model, history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, data):
    """Inference — returns (N,) numpy array."""
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        scores = model(data)
    return scores.detach().cpu().numpy()


def predict_with_uncertainty(model, data, n_forward: int = 20):
    """
    MC Dropout epistemic uncertainty.

    Returns:
        mean_scores: (N,)
        std_scores:  (N,)
    """
    device = next(model.parameters()).device
    data = data.to(device)
    preds = []
    for _ in range(n_forward):
        model.train()
        with torch.no_grad():
            preds.append(model(data, mc_dropout=True).detach().cpu().numpy())

    preds = np.array(preds)
    model.eval()
    return preds.mean(axis=0), preds.std(axis=0)
