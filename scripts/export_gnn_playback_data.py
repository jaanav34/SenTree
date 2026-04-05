"""Export GNN training history to JSON for the React playback app."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

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


def export_training_history(input_path: str, output_path: str) -> None:
    with np.load(input_path) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        edge_index = np.asarray(data["edge_index"], dtype=np.int32)
        target = np.asarray(data["target"], dtype=np.float32)
        predictions = np.asarray(data["predictions"], dtype=np.float32)
        loss = np.asarray(data["loss"], dtype=np.float32)
        learning_rate = np.asarray(data["learning_rate"], dtype=np.float32)

    epochs = np.arange(1, predictions.shape[0] + 1, dtype=np.int32)
    mean_risk = predictions.mean(axis=1).astype(np.float32)
    p95_risk = np.percentile(predictions, 95, axis=1).astype(np.float32)
    max_risk = predictions.max(axis=1).astype(np.float32)
    tail_threshold = float(np.percentile(target, 95))

    payload = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "source": input_path,
            "epochCount": int(predictions.shape[0]),
            "nodeCount": int(predictions.shape[1]),
            "edgeCount": int(edge_index.shape[1]),
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

    print(f"Exported GNN playback data to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GNN training history for the React playback app.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to gnn_training_history.npz")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output JSON file")
    args = parser.parse_args()

    export_training_history(args.input, args.output)


if __name__ == "__main__":
    main()
