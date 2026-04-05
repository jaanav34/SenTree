"""
Export intervention keys (and optional metadata) for Slurm job arrays.

Writes:
  - <out_dir>/intervention_keys.txt   (one key per line, stable order)
  - <out_dir>/intervention_names.json (key -> human name)

Typical usage (Gautschi):
  python scripts/export_intervention_keys.py --out-dir outputs/roi/risk_series
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

from src.simulation.interventions import INTERVENTIONS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/roi/risk_series")
    parser.add_argument(
        "--series-dir",
        default=None,
        help="Optional directory to scan for intervention_*.npz; if provided and any are found, keys are derived from filenames.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys: list[str] = []
    if args.series_dir:
        series_dir = Path(args.series_dir)
        if series_dir.exists():
            discovered = []
            for p in sorted(series_dir.glob("intervention_*.npz")):
                name = p.name
                if not name.startswith("intervention_") or not name.endswith(".npz"):
                    continue
                discovered.append(name[len("intervention_") : -len(".npz")])
            keys = discovered
    if not keys:
        keys = sorted(INTERVENTIONS.keys())
    (out_dir / "intervention_keys.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")

    names = {k: str(INTERVENTIONS[k].get("name", k)) for k in keys}
    (out_dir / "intervention_names.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(keys)} keys: {out_dir / 'intervention_keys.txt'}")
    print(f"Wrote names map: {out_dir / 'intervention_names.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
