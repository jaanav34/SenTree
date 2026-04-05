"""Export GNN training history to JSON for the React playback app."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

import numpy as np


DEFAULT_INPUT = "outputs/roi/gnn_training_history.npz"
DEFAULT_OUTPUT = "apps/gnn-playback/public/data/gnn_training_history.json"


def _to_rounded_list(array: np.ndarray, digits: int = 6):
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(np.float32), digits)
    return arr.tolist()

def _sample_nodes(
    positions: np.ndarray,
    target: np.ndarray,
    *,
    max_nodes: int,
    seed: int,
) -> np.ndarray:
    n_nodes = int(target.shape[0])
    if max_nodes <= 0 or max_nodes >= n_nodes:
        return np.arange(n_nodes, dtype=np.int32)

    rng = np.random.default_rng(seed)

    top_k = int(min(max_nodes // 5, 2000))
    if top_k <= 0:
        top_k = 1
    top_k = min(top_k, max_nodes, n_nodes)

    # Ensure we keep the highest-target nodes so the "tail-risk" highlights
    # remain meaningful even in downsampled playback mode.
    top_idx = np.argpartition(target, -top_k)[-top_k:].astype(np.int32)
    top_set = np.zeros(n_nodes, dtype=bool)
    top_set[top_idx] = True

    remaining = int(max_nodes - top_idx.size)
    if remaining > 0:
        pool = np.where(~top_set)[0]
        if pool.size > 0:
            pick = rng.choice(pool, size=min(remaining, pool.size), replace=False).astype(np.int32)
            sampled = np.concatenate([top_idx, pick], axis=0)
        else:
            sampled = top_idx
    else:
        sampled = top_idx

    # Deterministic ordering: roughly spatial (lat, lon) to keep playback stable.
    sampled = np.unique(sampled)
    lat = positions[sampled, 0]
    lon = positions[sampled, 1]
    order = np.lexsort((lon, lat))
    return sampled[order].astype(np.int32)


def _filter_edges(
    edge_index: np.ndarray,
    mapping: np.ndarray,
    *,
    max_edges: int,
    seed: int,
) -> np.ndarray:
    src = edge_index[0]
    dst = edge_index[1]
    src_new = mapping[src]
    dst_new = mapping[dst]
    mask = (src_new >= 0) & (dst_new >= 0)
    filtered_src = src_new[mask].astype(np.int32, copy=False)
    filtered_dst = dst_new[mask].astype(np.int32, copy=False)
    if filtered_src.size == 0:
        return np.zeros((2, 0), dtype=np.int32)

    if max_edges > 0 and filtered_src.size > max_edges:
        rng = np.random.default_rng(seed)
        idx = rng.choice(filtered_src.size, size=max_edges, replace=False)
        filtered_src = filtered_src[idx]
        filtered_dst = filtered_dst[idx]

    return np.stack([filtered_src, filtered_dst], axis=0).astype(np.int32)


def export_training_history(
    input_path: str,
    output_path: str,
    *,
    max_nodes: int,
    max_edges: int,
    seed: int,
) -> None:
    with np.load(input_path) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        edge_index = np.asarray(data["edge_index"], dtype=np.int32)
        target = np.asarray(data["target"], dtype=np.float32)
        predictions = np.asarray(data["predictions"], dtype=np.float32)
        loss = np.asarray(data["loss"], dtype=np.float32)
        learning_rate = np.asarray(data["learning_rate"], dtype=np.float32)

    epochs = np.arange(1, predictions.shape[0] + 1, dtype=np.int32)
    tail_threshold = float(np.percentile(target, 95))

    original_nodes = int(target.shape[0])
    original_edges = int(edge_index.shape[1]) if edge_index.ndim == 2 else int(edge_index.shape[0])

    sampled_idx = _sample_nodes(positions, target, max_nodes=max_nodes, seed=seed)
    mapping = np.full(target.shape[0], -1, dtype=np.int32)
    mapping[sampled_idx] = np.arange(sampled_idx.size, dtype=np.int32)

    positions = positions[sampled_idx]
    target = target[sampled_idx]
    predictions = predictions[:, sampled_idx]
    edge_index = _filter_edges(edge_index, mapping, max_edges=max_edges, seed=seed)

    mean_risk = predictions.mean(axis=1).astype(np.float32)
    p95_risk = np.percentile(predictions, 95, axis=1).astype(np.float32)
    max_risk = predictions.max(axis=1).astype(np.float32)

    payload = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "source": input_path,
            "epochCount": int(predictions.shape[0]),
            "nodeCount": int(predictions.shape[1]),
            "edgeCount": int(edge_index.shape[1]),
            "sampledFromNodes": int(mapping.shape[0]),
            "sampleMaxNodes": int(max_nodes),
            "sampleMaxEdges": int(max_edges),
            "sampleSeed": int(seed),
        },
        "epochs": epochs.tolist(),
        "positions": _to_rounded_list(positions),
        "edgeIndex": edge_index.tolist(),
        "target": _to_rounded_list(target),
        "predictions": _to_rounded_list(predictions),
        "loss": _to_rounded_list(loss),
        "learningRate": _to_rounded_list(learning_rate),
        "meanRisk": _to_rounded_list(mean_risk),
        "p95Risk": _to_rounded_list(p95_risk),
        "maxRisk": _to_rounded_list(max_risk),
        "tailThreshold": round(tail_threshold, 6),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    try:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"Exported GNN playback data to {output_path} "
            f"(nodes={predictions.shape[1]}/{original_nodes}, edges={edge_index.shape[1]}/{original_edges}, size={size_mb:.1f}MB)"
        )
    except Exception:
        print(f"Exported GNN playback data to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GNN training history for the React playback app.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to gnn_training_history.npz")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output JSON file")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=int(os.environ.get("SENTREE_PLAYBACK_MAX_NODES", "5000")),
        help="Downsample to at most this many nodes (default: 5000; set 0 for full export).",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=int(os.environ.get("SENTREE_PLAYBACK_MAX_EDGES", "200000")),
        help="Keep at most this many edges after filtering to sampled nodes (default: 200000; set 0 for all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SENTREE_PLAYBACK_SEED", "0")),
        help="RNG seed for sampling (default: 0).",
    )
    args = parser.parse_args()

    export_training_history(
        args.input,
        args.output,
        max_nodes=int(args.max_nodes),
        max_edges=int(args.max_edges),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
