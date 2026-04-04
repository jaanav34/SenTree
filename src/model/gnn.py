"""Minimal 2-layer GCN for risk propagation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ClimateRiskGNN(nn.Module):
    """
    2-layer GCN.
    Input: node features (N, F)
    Output: risk score per node (N,)
    """

    def __init__(self, in_channels=5, hidden_channels=32, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        risk = torch.sigmoid(self.head(x))
        return risk.squeeze(-1)


def train_gnn(model, data, target_scores, epochs=50, lr=0.01):
    """Quick training using tail-risk scores as pseudo-labels."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target = torch.tensor(target_scores, dtype=torch.float32)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def predict(model, data):
    """Run inference, return risk scores as numpy."""
    model.eval()
    with torch.no_grad():
        scores = model(data)
    return scores.numpy()
