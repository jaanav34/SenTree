"""
Summarize per-intervention comparison render timings.

Reads JSON files written by scripts/render_comparison_from_npz.py:
  outputs/roi/risk_series/render_timings/comparison_<key>.json

Usage:
  python scripts/summarize_render_timings.py
"""

from __future__ import annotations

import json
from pathlib import Path


def _format_seconds(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.2f}m"
    return f"{seconds:.2f}s"


def main() -> int:
    timings_dir = Path("outputs/roi/risk_series/render_timings")
    if not timings_dir.exists():
        print("No render timings found at outputs/roi/risk_series/render_timings")
        return 1

    rows = []
    for p in sorted(timings_dir.glob("comparison_*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue

    if not rows:
        print("No readable timing JSON files found.")
        return 1

    rows.sort(key=lambda r: float(r.get("seconds", 0.0)), reverse=True)
    total = sum(float(r.get("seconds", 0.0)) for r in rows)
    max_one = float(rows[0].get("seconds", 0.0))
    n = len(rows)

    print("=" * 70)
    print(f"Comparison render timings: {n} videos")
    print("=" * 70)
    print(f"Total CPU-seconds (sum): {_format_seconds(total)}")
    print(f"Slowest single video:    {_format_seconds(max_one)}  ({rows[0].get('key')})")
    print("-" * 70)
    for r in rows[:10]:
        print(f"{r.get('key','?'):<30} {_format_seconds(float(r.get('seconds',0.0))):>10}  {r.get('name','')}")
    if n > 10:
        print(f"... {n-10} more")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

