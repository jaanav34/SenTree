"""Generate heatmap videos from climate/risk data."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .downscale import downscale_grid
import os
import shutil
from pathlib import Path


def _save_animation(ani, output_path: str, *, fps: int, dpi: int = 100) -> None:
    ext = Path(output_path).suffix.lower()

    if ext == ".mp4":
        ffmpeg_env = os.environ.get("SENTREE_FFMPEG_PATH") or os.environ.get("SENTREE_FFMPEG")
        if ffmpeg_env:
            ffmpeg_path = Path(ffmpeg_env)
            try:
                if not ffmpeg_path.exists():
                    raise RuntimeError(f"SENTREE_FFMPEG_PATH points to a missing file: {ffmpeg_path}")
            except PermissionError:
                # Some environments (sandboxes / locked-down folders) may deny stat() even though
                # the executable is runnable by the user. Let Matplotlib try to use it.
                pass
            matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path)

        if shutil.which("ffmpeg") is None and not ffmpeg_env:
            raise RuntimeError(
                "Cannot write .mp4 because `ffmpeg` is not installed or not on PATH.\n\n"
                "Windows (recommended):  winget install Gyan.FFmpeg\n"
                "If it's installed but not on PATH, set e.g.:\n"
                "  $env:SENTREE_FFMPEG_PATH = 'C:\\path\\to\\ffmpeg.exe'\n"
                "Then restart your terminal and verify:  ffmpeg -version\n\n"
                "Alternative: change output to a .gif and re-run."
            )
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(output_path, writer=writer, dpi=dpi)
        return

    if ext == ".gif":
        ani.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        return

    raise ValueError(f"Unsupported video extension: {ext} (expected .mp4 or .gif)")


def _risk_stats_series(risk_series, years=None):
    if years is None:
        years = np.arange(2015, 2015 + len(risk_series))
    years = np.asarray(years)[:len(risk_series)]
    means = np.array([r.mean() for r in risk_series], dtype=np.float32)
    p95 = np.array([np.percentile(r, 95) for r in risk_series], dtype=np.float32)
    mx = np.array([r.max() for r in risk_series], dtype=np.float32)
    return years, means, p95, mx


def render_risk_video(risk_series, lats, lons, output_path, title='Risk Heatmap',
                      fps=4, scale_factor=8, cmap='YlOrRd', year_labels=None):
    """Render a time series of risk grids as an MP4 video."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])

    hires = downscale_grid(risk_series[0], scale_factor=scale_factor)
    vmin = min(r.min() for r in risk_series)
    vmax = max(r.max() for r in risk_series)

    im = ax.imshow(hires, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   extent=[lons[0], lons[-1], lats[0], lats[-1]], aspect='auto')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='Risk Score')
    _yl = year_labels if year_labels is not None else np.arange(2015, 2015 + len(risk_series))
    title_text = ax.set_title(f'{title} — Year {_yl[0]}')

    years, means, p95, mx = _risk_stats_series(risk_series, years=year_labels)
    ax_ts.set_title("Risk Summary")
    ax_ts.set_xlabel("Year")
    ax_ts.set_ylabel("Risk")
    ax_ts.grid(True, alpha=0.25)
    (line_mean,) = ax_ts.plot([], [], label="Mean", color="#1f77b4", linewidth=2)
    (line_p95,) = ax_ts.plot([], [], label="P95", color="#ff7f0e", linewidth=2)
    (line_max,) = ax_ts.plot([], [], label="Max", color="#d62728", linewidth=2, alpha=0.8)
    (marker,) = ax_ts.plot([], [], marker="o", color="black", markersize=5, linestyle="")
    ax_ts.set_xlim(years[0], years[-1])
    ax_ts.set_ylim(float(vmin), float(vmax))
    ax_ts.legend(loc="upper left")

    def update(frame):
        hires = downscale_grid(risk_series[frame], scale_factor=scale_factor)
        im.set_data(hires)
        title_text.set_text(f'{title} — Year {_yl[frame]}')

        x = years[: frame + 1]
        line_mean.set_data(x, means[: frame + 1])
        line_p95.set_data(x, p95[: frame + 1])
        line_max.set_data(x, mx[: frame + 1])
        marker.set_data([years[frame]], [means[frame]])

        return [im, title_text, line_mean, line_p95, line_max, marker]

    ani = animation.FuncAnimation(fig, update, frames=len(risk_series),
                                   interval=1000//fps, blit=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _save_animation(ani, output_path, fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved video: {output_path}")
    return output_path


def render_comparison_video(baseline_series, intervention_series, lats, lons,
                            output_path, intervention_name='Intervention',
                            fps=4, scale_factor=8, year_labels=None):
    """Side-by-side comparison: baseline vs intervention."""
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[0, 2])

    vmin = min(min(r.min() for r in baseline_series),
               min(r.min() for r in intervention_series))
    vmax = max(max(r.max() for r in baseline_series),
               max(r.max() for r in intervention_series))

    extent = [lons[0], lons[-1], lats[0], lats[-1]]

    h1 = downscale_grid(baseline_series[0], scale_factor=scale_factor)
    h2 = downscale_grid(intervention_series[0], scale_factor=scale_factor)

    im1 = ax1.imshow(h1, cmap='YlOrRd', origin='lower', vmin=vmin, vmax=vmax,
                     extent=extent, aspect='auto')
    im2 = ax2.imshow(h2, cmap='YlOrRd', origin='lower', vmin=vmin, vmax=vmax,
                     extent=extent, aspect='auto')

    ax1.set_title('Baseline Risk')
    ax2.set_title(f'With {intervention_name}')
    plt.colorbar(im1, ax=ax1, label='Risk')
    plt.colorbar(im2, ax=ax2, label='Risk')

    _yl = year_labels if year_labels is not None else np.arange(2015, 2015 + len(baseline_series))
    year_text = fig.suptitle(f'Year {_yl[0]}', fontsize=14, fontweight='bold')

    years, b_mean, b_p95, _b_max = _risk_stats_series(baseline_series, years=year_labels)
    _years2, i_mean, i_p95, _i_max = _risk_stats_series(intervention_series, years=year_labels)

    ax_ts.set_title("Quantitative Comparison")
    ax_ts.set_xlabel("Year")
    ax_ts.set_ylabel("Risk")
    ax_ts.grid(True, alpha=0.25)
    (line_b,) = ax_ts.plot([], [], label="Baseline mean", color="#1f77b4", linewidth=2)
    (line_i,) = ax_ts.plot([], [], label="Intervention mean", color="#2ca02c", linewidth=2)

    ax_delta = ax_ts.twinx()
    ax_delta.set_ylabel("Delta (B - I)")
    delta = b_mean - i_mean
    dmax = float(max(abs(delta.min()), abs(delta.max()), 1e-6))
    ax_delta.set_ylim(-dmax, dmax)
    (line_delta,) = ax_delta.plot([], [], label="Delta (B-I)", color="#d62728", linewidth=1.8, linestyle="--")
    ax_ts.set_xlim(years[0], years[-1])
    ax_ts.set_ylim(float(vmin), float(vmax))
    lines = [line_b, line_i, line_delta]
    ax_ts.legend(lines, [l.get_label() for l in lines], loc="upper left")

    def update(frame):
        im1.set_data(downscale_grid(baseline_series[frame], scale_factor=scale_factor))
        im2.set_data(downscale_grid(intervention_series[frame], scale_factor=scale_factor))
        year_text.set_text(f'Year {_yl[frame]}')

        x = years[: frame + 1]
        line_b.set_data(x, b_mean[: frame + 1])
        line_i.set_data(x, i_mean[: frame + 1])
        line_delta.set_data(x, (b_mean - i_mean)[: frame + 1])

        return [im1, im2, year_text, line_b, line_i, line_delta]

    ani = animation.FuncAnimation(fig, update, frames=len(baseline_series),
                                   interval=1000//fps, blit=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _save_animation(ani, output_path, fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved comparison video: {output_path}")
    return output_path


def render_tail_risk_video(risk_series, flags_series, lats, lons, output_path,
                           fps=4, scale_factor=8, year_labels=None):
    """Risk heatmap with tail-risk nodes highlighted."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    extent = [lons[0], lons[-1], lats[0], lats[-1]]

    hires = downscale_grid(risk_series[0], scale_factor=scale_factor)
    vmin = min(r.min() for r in risk_series)
    vmax = max(r.max() for r in risk_series)

    im = ax.imshow(hires, cmap='YlOrRd', origin='lower', vmin=vmin, vmax=vmax,
                   extent=extent, aspect='auto')

    nlat, nlon = risk_series[0].shape
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    lat_flat, lon_flat = lat_grid.flatten(), lon_grid.flatten()

    flagged = flags_series[0].flatten()
    scatter = ax.scatter(lon_flat[flagged], lat_flat[flagged],
                         c='red', s=50, marker='X', label='Tail-Risk Node', zorder=5)

    ax.legend(loc='upper left')
    plt.colorbar(im, ax=ax, label='Risk Score')
    _yl = year_labels if year_labels is not None else np.arange(2015, 2015 + len(risk_series))
    title_text = ax.set_title(f'Tail-Risk Escalation — Year {_yl[0]}')

    years, means, p95, mx = _risk_stats_series(risk_series, years=year_labels)
    flagged_counts = np.array([int(f.sum()) for f in flags_series], dtype=np.int32)

    ax_ts.set_title("Tail-Risk Summary")
    ax_ts.set_xlabel("Year")
    ax_ts.grid(True, alpha=0.25)
    ax_ts_risk = ax_ts
    ax_ts_count = ax_ts.twinx()

    ax_ts_risk.set_ylabel("Risk")
    ax_ts_count.set_ylabel("Flagged nodes")

    (line_p95,) = ax_ts_risk.plot([], [], label="Risk P95", color="#ff7f0e", linewidth=2)
    (line_mean,) = ax_ts_risk.plot([], [], label="Risk mean", color="#1f77b4", linewidth=2)
    (bar_count,) = ax_ts_count.plot([], [], label="Flagged count", color="#d62728", linewidth=2, linestyle="--")

    ax_ts_risk.set_xlim(years[0], years[-1])
    ax_ts_risk.set_ylim(float(vmin), float(vmax))
    ax_ts_count.set_ylim(0, max(1, int(flagged_counts.max())))

    # Merge legends
    lines = [line_p95, line_mean, bar_count]
    ax_ts_risk.legend(lines, [l.get_label() for l in lines], loc="upper left")

    def update(frame):
        hires = downscale_grid(risk_series[frame], scale_factor=scale_factor)
        im.set_data(hires)
        flagged = flags_series[frame].flatten()
        offsets = np.column_stack([lon_flat[flagged], lat_flat[flagged]])
        scatter.set_offsets(offsets if len(offsets) > 0 else np.empty((0, 2)))
        title_text.set_text(f'Tail-Risk Escalation — Year {_yl[frame]}')

        x = years[: frame + 1]
        line_p95.set_data(x, p95[: frame + 1])
        line_mean.set_data(x, means[: frame + 1])
        bar_count.set_data(x, flagged_counts[: frame + 1])

        return [im, scatter, title_text, line_p95, line_mean, bar_count]

    ani = animation.FuncAnimation(fig, update, frames=len(risk_series),
                                   interval=1000//fps, blit=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _save_animation(ani, output_path, fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved tail-risk video: {output_path}")
    return output_path


def render_tail_risk_map(value_grid, flags_grid, lats, lons, output_path,
                        title='Resilience Opportunity Map', label='ROI Potential (Risk Reduction)',
                        scale_factor=8):
    """Save a coordinate-accurate map highlighting ROI potential and target nodes."""
    fig, ax = plt.subplots(figsize=(14, 8))
    extent = [lons[0], lons[-1], lats[0], lats[-1]]

    # Background: Resilience Opportunity (Green = high reduction potential)
    hires = downscale_grid(value_grid, scale_factor=scale_factor)
    im = ax.imshow(hires, cmap='Greens', origin='lower',
                   extent=extent, aspect='auto')

    # Add Coordinate Grid for exact location identification
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_xticks(np.linspace(lons[0], lons[-1], 10))
    ax.set_yticks(np.linspace(lats[0], lats[-1], 10))

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    lat_flat, lon_flat = lat_grid.flatten(), lon_grid.flatten()
    flagged = flags_grid.flatten()

    # Scatter: Target high-risk nodes (X marks the spot for highest ROI)
    ax.scatter(lon_flat[flagged], lat_flat[flagged],
               c='red', s=70, marker='X', label='High-Risk / High-ROI Target', 
               edgecolors='white', linewidths=0.5, zorder=5)

    # Precise Coordinate Annotation for Top 3 ROI Nodes
    top_indices = np.argsort(value_grid.flatten())[-3:]
    for idx in top_indices:
        ax.annotate(f"Target: ({lon_flat[idx]:.2f}E, {lat_flat[idx]:.2f}N)",
                    (lon_flat[idx], lat_flat[idx]),
                    textcoords="offset points", xytext=(10,10), 
                    ha='left', fontsize=9, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    plt.colorbar(im, ax=ax, label=label)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Success: Resilience Opportunity Map saved to {output_path}")
    return output_path


# Köppen-Geiger color mapping for visualization
KG_COLORS = {
    1: '#0000FF', 2: '#0077FF', 3: '#44AAFF', 4: '#77CCFF', # Group A: Tropical (Blue)
    5: '#FF0000', 6: '#FF7777', 7: '#FFAA00', 8: '#FFCC00', # Group B: Dry (Red/Orange)
    9: '#00FF00', 10: '#33FF33', 11: '#66FF66', 12: '#99FF99', 13: '#CCFFCC', 14: '#00AA00', # Group C: Temperate (Green)
    15: '#AAFF00', 16: '#DDFF00', 17: '#FFFF00',
    18: '#AA00FF', 19: '#CC00FF', 20: '#EE00FF', 21: '#FF00FF', # Group D: Continental (Purple/Magenta)
    22: '#AA77FF', 23: '#CC99FF', 24: '#EECCFF', 25: '#FFCCFF',
    26: '#7700AA', 27: '#9900CC', 28: '#CC00EE', 29: '#EE00FF',
    30: '#AAAAAA', 31: '#DDDDDD', # Group E: Polar (Grey)
    0: '#000000' # Unknown
}

def render_kg_video(kg_series, lats, lons, output_path, fps=4, year_labels=None):
    """Render a time series of Köppen-Geiger classifications."""
    from src.data.koppen_geiger import KG_LABELS
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12, 8))
    extent = [lons[0], lons[-1], lats[0], lats[-1]]

    # Create discrete colormap for only used values
    unique_vals = sorted(np.unique(kg_series))
    if 0 not in unique_vals: unique_vals = [0] + unique_vals
    
    colors = [KG_COLORS.get(v, '#FFFFFF') for v in unique_vals]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(unique_vals) + 1) - 0.5, len(unique_vals))

    # Create a mapping from KG value to 0..N-1 for imshow
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    mapped_kg = np.vectorize(val_to_idx.get)(kg_series[0])

    im = ax.imshow(mapped_kg, cmap=cmap, norm=norm, origin='lower',
                   extent=extent, aspect='auto', interpolation='nearest')

    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    
    current_year = year_labels[0] if year_labels is not None else 2015
    title_text = ax.set_title(f'Köppen-Geiger Climate Classification — Year {current_year}')

    # Legend for present classes
    legend_elements = [Patch(facecolor=KG_COLORS.get(v), label=f"{v}: {KG_LABELS.get(v, 'Unknown')}")
                      for v in unique_vals if v != 0]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Climate Zones")

    def update(frame):
        mapped = np.vectorize(val_to_idx.get)(kg_series[frame])
        im.set_data(mapped)
        year = year_labels[frame] if year_labels is not None else 2015 + frame
        title_text.set_text(f'Köppen-Geiger Climate Classification — Year {year}')
        return [im, title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(kg_series),
                                   interval=1000//fps, blit=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _save_animation(ani, output_path, fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved Köppen-Geiger video: {output_path}")
    return output_path
