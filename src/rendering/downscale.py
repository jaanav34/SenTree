"""Fake generative downscaling via interpolation + smoothing."""
import numpy as np
from scipy.ndimage import zoom, gaussian_filter


def downscale_grid(coarse_grid, scale_factor=8, sigma=1.5):
    """
    Upsample a coarse grid to fake high-resolution output.

    Args:
        coarse_grid: (nlat, nlon) numpy array
        scale_factor: upsampling factor (8x = 2deg -> 0.25deg equivalent)
        sigma: Gaussian smoothing to remove blocky artifacts

    Returns:
        hires_grid: (nlat*scale, nlon*scale) numpy array
    """
    hires = zoom(coarse_grid, scale_factor, order=3)
    hires = gaussian_filter(hires, sigma=sigma)
    return hires


def downscale_timeseries(data_3d, scale_factor=8, sigma=1.5):
    """
    Downscale entire time series.
    Input: (T, nlat, nlon)
    Output: (T, nlat*scale, nlon*scale)
    """
    T = data_3d.shape[0]
    first = downscale_grid(data_3d[0], scale_factor, sigma)
    result = np.zeros((T, *first.shape))
    result[0] = first

    for t in range(1, T):
        result[t] = downscale_grid(data_3d[t], scale_factor, sigma)

    return result
