"""Convert grid cells to PyTorch Geometric graph."""
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph


def build_climate_graph(features, positions, k=8):
    """
    Build a PyG Data object from node features and positions.

    Args:
        features: (N, F) numpy array
        positions: (N, 2) numpy array — (lat, lon)
        k: number of nearest neighbors

    Returns:
        PyG Data object
    """
    N = features.shape[0]

    adj = kneighbors_graph(positions, n_neighbors=k, mode='connectivity', include_self=False)
    edge_index = np.array(adj.nonzero())

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, pos=pos)
    data.num_nodes = N

    return data


def build_grid_adjacency(nlat, nlon):
    """8-connected grid adjacency. Fallback if KNN is slow."""
    edges = []
    for i in range(nlat):
        for j in range(nlon):
            node = i * nlon + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nlat and 0 <= nj < nlon:
                    edges.append([node, ni * nlon + nj])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()
