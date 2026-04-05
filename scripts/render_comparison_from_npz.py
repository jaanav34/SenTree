"""
Render a comparison video from saved risk series NPZ files.

This is intended for Slurm job arrays so you can render many intervention
videos in parallel on CPU nodes, without re-running the full pipeline.

Inputs expected (written by scripts/run_pipeline.py when
SENTREE_SAVE_RISK_SERIES_NPZ=1):
  - outputs/roi/risk_series/baseline.npz (baseline, years, lats, lons)
  - outputs/roi/risk_series/intervention_<key>.npz (intervention)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

from src.rendering.render_video import render_comparison_video
from src.simulation.interventions import INTERVENTIONS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Intervention key (e.g., mangrove_restoration)")
    parser.add_argument("--name", default=None, help="Human name for titles (optional)")
    parser.add_argument(
        "--names-json",
        default="outputs/roi/risk_series/intervention_names.json",
        help="Optional JSON mapping key->name; used if --name omitted.",
    )
    parser.add_argument("--baseline", default="outputs/roi/risk_series/baseline.npz")
    parser.add_argument("--intervention", default=None, help="NPZ path; defaults to outputs/roi/risk_series/intervention_<key>.npz")
    parser.add_argument("--out", default=None, help="Output mp4 path; defaults to outputs/videos/comparison_<key>.mp4")
    parser.add_argument("--fps", type=int, default=int(__import__("os").environ.get("SENTREE_RENDER_FPS", "4")))
    parser.add_argument("--scale-factor", type=int, default=int(__import__("os").environ.get("SENTREE_RENDER_SCALE_FACTOR", "8")))
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    intervention_path = Path(args.intervention) if args.intervention else Path(f"outputs/roi/risk_series/intervention_{args.key}.npz")
    out_path = Path(args.out) if args.out else Path(f"outputs/videos/comparison_{args.key}.mp4")

    name = args.name
    if not name:
        names_path = Path(args.names_json)
        if names_path.exists():
            try:
                mapping = json.loads(names_path.read_text(encoding="utf-8"))
                name = mapping.get(args.key) or args.key
            except Exception:
                name = args.key
        else:
            name = str(INTERVENTIONS.get(args.key, {}).get("name", args.key))

    b = np.load(baseline_path, allow_pickle=False)
    i = np.load(intervention_path, allow_pickle=False)

    baseline = b["baseline"].astype(np.float32)  # (T, nlat, nlon)
    intervention = i["intervention"].astype(np.float32)
    years = b["years"].astype(int)
    lats = b["lats"].astype(float)
    lons = b["lons"].astype(float)

    baseline_series = [baseline[t] for t in range(baseline.shape[0])]
    intervention_series = [intervention[t] for t in range(intervention.shape[0])]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    render_comparison_video(
        baseline_series,
        intervention_series,
        lats,
        lons,
        str(out_path),
        intervention_name=name,
        year_labels=years,
        fps=args.fps,
        scale_factor=args.scale_factor,
    )
    dt = float(time.perf_counter() - t0)

    # Per-task timing artifact for Slurm arrays.
    timings_dir = Path("outputs/roi/risk_series/render_timings")
    timings_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": args.key,
        "name": name,
        "output": str(out_path),
        "seconds": dt,
        "fps": int(args.fps),
        "scale_factor": int(args.scale_factor),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    (timings_dir / f"comparison_{args.key}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Timing: {dt:.2f}s (wrote outputs/roi/risk_series/render_timings/comparison_{args.key}.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
