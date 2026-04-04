"""Generate heatmap videos from climate/risk data."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .downscale import downscale_grid
import os


def render_risk_video(risk_series, lats, lons, output_path, title='Risk Heatmap',
                      fps=4, scale_factor=8, cmap='YlOrRd'):
    """Render a time series of risk grids as an MP4 video."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    hires = downscale_grid(risk_series[0], scale_factor=scale_factor)
    vmin = min(r.min() for r in risk_series)
    vmax = max(r.max() for r in risk_series)

    im = ax.imshow(hires, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   extent=[lons[0], lons[-1], lats[0], lats[-1]], aspect='auto')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='Risk Score')
    title_text = ax.set_title(f'{title} — Year 2015')

    def update(frame):
        hires = downscale_grid(risk_series[frame], scale_factor=scale_factor)
        im.set_data(hires)
        title_text.set_text(f'{title} — Year {2015 + frame}')
        return [im, title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(risk_series),
                                   interval=1000//fps, blit=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved video: {output_path}")
    return output_path


def render_comparison_video(baseline_series, intervention_series, lats, lons,
                            output_path, intervention_name='Intervention',
                            fps=4, scale_factor=8):
    """Side-by-side comparison: baseline vs intervention."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

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

    year_text = fig.suptitle('Year 2015', fontsize=14, fontweight='bold')

    def update(frame):
        im1.set_data(downscale_grid(baseline_series[frame], scale_factor=scale_factor))
        im2.set_data(downscale_grid(intervention_series[frame], scale_factor=scale_factor))
        year_text.set_text(f'Year {2015 + frame}')
        return [im1, im2, year_text]

    ani = animation.FuncAnimation(fig, update, frames=len(baseline_series),
                                   interval=1000//fps, blit=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved comparison video: {output_path}")
    return output_path


def render_tail_risk_video(risk_series, flags_series, lats, lons, output_path,
                           fps=4, scale_factor=8):
    """Risk heatmap with tail-risk nodes highlighted."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
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
    title_text = ax.set_title('Tail-Risk Escalation — Year 2015')

    def update(frame):
        hires = downscale_grid(risk_series[frame], scale_factor=scale_factor)
        im.set_data(hires)
        flagged = flags_series[frame].flatten()
        offsets = np.column_stack([lon_flat[flagged], lat_flat[flagged]])
        scatter.set_offsets(offsets if len(offsets) > 0 else np.empty((0, 2)))
        title_text.set_text(f'Tail-Risk Escalation — Year {2015 + frame}')
        return [im, scatter, title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(risk_series),
                                   interval=1000//fps, blit=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved tail-risk video: {output_path}")
    return output_path
