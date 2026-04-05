"""
Render a "mega" video from saved risk series NPZ files.

Uses risk series written by scripts/run_pipeline.py when:
  SENTREE_SAVE_RISK_SERIES_NPZ=1

Modes:
  - grid (default): show all interventions simultaneously in a grid of mini-panels
  - cycle: show one intervention at a time (legacy / lighter)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.rendering.render_video import _save_animation
from src.simulation.interventions import INTERVENTIONS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["grid", "cycle"], default="grid")
    parser.add_argument(
        "--keys",
        default=None,
        help="Optional path to intervention_keys.txt. If omitted/missing, keys are discovered from NPZs in --series-dir or from INTERVENTIONS.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Optional path to intervention_names.json. If omitted/missing, names are taken from INTERVENTIONS.",
    )
    parser.add_argument("--baseline", default="outputs/roi/risk_series/baseline.npz")
    parser.add_argument("--series-dir", default="outputs/roi/risk_series")
    parser.add_argument("--out", default="outputs/videos/interventions_megavideo.mp4")
    parser.add_argument("--fps", type=int, default=int(os.environ.get("SENTREE_RENDER_FPS", "4")))
    parser.add_argument("--seconds-per", type=float, default=float(os.environ.get("SENTREE_MEGA_SECONDS_PER", "1.2")))
    parser.add_argument("--year-idx", type=int, default=-1, help="Year index for single-frame render (default: last).")
    parser.add_argument("--animate-years", action="store_true", help="Animate through all years (heavier).")
    parser.add_argument("--hold-seconds", type=float, default=2.0, help="For single-frame mode, hold this many seconds.")
    parser.add_argument("--ncols", type=int, default=6, help="Grid columns for mode=grid.")
    parser.add_argument("--cmap", default="YlOrRd")
    args = parser.parse_args()

    keys: list[str] = []
    keys_path = Path(args.keys) if args.keys else None
    if keys_path and keys_path.exists():
        keys = [k.strip() for k in keys_path.read_text(encoding="utf-8").splitlines() if k.strip()]
    else:
        series_dir = Path(args.series_dir)
        if series_dir.exists():
            npz_keys = []
            for p in sorted(series_dir.glob("intervention_*.npz")):
                name = p.name
                if not name.startswith("intervention_") or not name.endswith(".npz"):
                    continue
                npz_keys.append(name[len("intervention_") : -len(".npz")])
            if npz_keys:
                keys = npz_keys
                print(f"Discovered {len(keys)} interventions from NPZs in {series_dir}.")
        if not keys:
            keys = sorted(INTERVENTIONS.keys())
            msg = f"Using INTERVENTIONS from code ({len(keys)} keys)"
            if keys_path:
                msg = f"Keys file not found ({keys_path}); {msg}."
            else:
                msg = f"No keys file provided; {msg}."
            print(msg)

    if not keys:
        raise SystemExit("No intervention keys available (no NPZs found and INTERVENTIONS is empty).")

    names = {}
    names_path = Path(args.names) if args.names else None
    if names_path and names_path.exists():
        try:
            names = json.loads(names_path.read_text(encoding="utf-8"))
        except Exception:
            names = {}
    if not names:
        names = {k: str(INTERVENTIONS.get(k, {}).get("name", k)) for k in keys}

    b = np.load(Path(args.baseline), allow_pickle=False)
    baseline = b["baseline"].astype(np.float32)
    years = b["years"].astype(int)
    lats = b["lats"].astype(float)
    lons = b["lons"].astype(float)

    extent = [float(lons[0]), float(lons[-1]), float(lats[0]), float(lats[-1])]

    T = int(baseline.shape[0])
    if args.animate_years:
        frame_year_indices = list(range(T))
    else:
        frame_year_indices = [args.year_idx if args.year_idx >= 0 else (T - 1)]

    # Preload interventions series (keeps render deterministic/fast; uses RAM).
    interventions = {}
    missing = []
    for key in keys:
        p = Path(args.series_dir) / f"intervention_{key}.npz"
        if not p.exists():
            missing.append(key)
            continue
        interventions[key] = np.load(p, allow_pickle=False)["intervention"].astype(np.float32)

    if missing:
        print(f"Skipping {len(missing)} interventions with no NPZ present under {args.series_dir}.")
        keys = [k for k in keys if k in interventions]

    if not keys:
        raise SystemExit(f"No intervention NPZs found under {args.series_dir}; cannot render mega video.")

    # Baseline scale fixed for visual consistency.
    base_vmin = float(np.nanmin(baseline))
    base_vmax = float(np.nanmax(baseline))

    if args.mode == "cycle":
        # --- Cycle mode (legacy): 2 panels, intervention changes each block ---
        t_idx = int(frame_year_indices[-1])
        year = int(years[t_idx]) if years.size else (2015 + t_idx)
        base_grid = baseline[t_idx]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        im0 = ax0.imshow(base_grid, origin="lower", extent=extent, cmap=args.cmap, vmin=base_vmin, vmax=base_vmax, aspect="auto")
        ax0.set_title(f"Baseline Risk — {year}")
        ax0.set_xlabel("Lon")
        ax0.set_ylabel("Lat")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        first_key = keys[0]
        delta0 = np.clip(base_grid - interventions[first_key][t_idx], 0.0, None)
        dmax = float(np.nanpercentile(delta0, 99.5)) if np.isfinite(delta0).any() else 1.0
        im1 = ax1.imshow(delta0, origin="lower", extent=extent, cmap="Greens", vmin=0.0, vmax=max(dmax, 1e-6), aspect="auto")
        ax1.set_title(f"Risk Reduction — {names.get(first_key, first_key)}")
        ax1.set_xlabel("Lon")
        ax1.set_ylabel("Lat")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Baseline - Intervention")

        frames_per = max(1, int(round(args.seconds_per * args.fps)))
        n_frames = frames_per * len(keys)

        def update(frame: int):
            key_idx = int(frame // frames_per)
            key = keys[key_idx]
            delta = np.clip(base_grid - interventions[key][t_idx], 0.0, None)
            im1.set_data(delta)
            ax1.set_title(f"Risk Reduction — {names.get(key, key)}")
            return [im1]

        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // max(1, args.fps), blit=False)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_animation(ani, str(out_path), fps=args.fps, dpi=100)
        plt.close(fig)
        print(f"Saved mega video (cycle mode): {out_path}")
        return 0

    # --- Grid mode: baseline + all deltas displayed simultaneously ---
    ncols = max(1, int(args.ncols))
    n_panels = 1 + len(keys)  # baseline + each intervention delta
    nrows = int(math.ceil(n_panels / ncols))

    fig_w = max(16.0, 3.0 * ncols)
    fig_h = max(9.0, 2.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    # Create baseline panel (index 0)
    ax_base = axes[0]
    t0 = int(frame_year_indices[0])
    base0 = baseline[t0]
    year0 = int(years[t0]) if years.size else (2015 + t0)
    im_base = ax_base.imshow(base0, origin="lower", extent=extent, cmap=args.cmap, vmin=base_vmin, vmax=base_vmax, aspect="auto")
    ax_base.set_title(f"Baseline — {year0}", fontsize=10, loc="left")
    ax_base.set_xticks([])
    ax_base.set_yticks([])

    # Determine a stable delta color scale.
    sample_t = int(frame_year_indices[0])
    sample_deltas = []
    for key in keys[: min(8, len(keys))]:
        sample_deltas.append(np.clip(baseline[sample_t] - interventions[key][sample_t], 0.0, None))
    stacked = np.stack(sample_deltas, axis=0) if sample_deltas else np.zeros((1,) + baseline[sample_t].shape, dtype=np.float32)
    delta_vmax = float(np.nanpercentile(stacked, 99.5))
    delta_vmax = max(delta_vmax, 1e-6)

    ims_delta = []
    for idx, key in enumerate(keys, start=1):
        ax = axes[idx]
        delta = np.clip(baseline[t0] - interventions[key][t0], 0.0, None)
        im = ax.imshow(delta, origin="lower", extent=extent, cmap="Greens", vmin=0.0, vmax=delta_vmax, aspect="auto")
        ax.set_title(names.get(key, key), fontsize=8, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ims_delta.append((key, im, ax))

    # Hide any unused panels
    for ax in axes[n_panels:]:
        ax.axis("off")

    # Add compact shared colorbars
    try:
        fig.colorbar(im_base, ax=ax_base, fraction=0.03, pad=0.01)
    except Exception:
        pass
    try:
        fig.colorbar(ims_delta[0][1], ax=[ax for _, _, ax in ims_delta], fraction=0.02, pad=0.01)
    except Exception:
        pass

    if len(frame_year_indices) == 1:
        # Single-frame video: repeat the same frame for hold-seconds.
        n_frames = max(1, int(round(args.hold_seconds * args.fps)))
        frame_year_indices = frame_year_indices * n_frames

    def update(frame: int):
        t_idx = int(frame_year_indices[frame])
        yr = int(years[t_idx]) if years.size else (2015 + t_idx)
        base_grid = baseline[t_idx]
        im_base.set_data(base_grid)
        ax_base.set_title(f"Baseline — {yr}", fontsize=10, loc="left")
        for key, im, _ax in ims_delta:
            im.set_data(np.clip(base_grid - interventions[key][t_idx], 0.0, None))
        return [im_base] + [im for _, im, _ in ims_delta]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_year_indices), interval=1000 // max(1, args.fps), blit=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_animation(ani, str(out_path), fps=args.fps, dpi=100)
    plt.close(fig)
    print(f"Saved mega video (grid mode): {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
