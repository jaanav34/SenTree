"""Resilience ROI calculation with uncertainty penalty."""
import numpy as np


def compute_roi(baseline_risk, intervention_risk, cost, gdp_flat, pop_flat,
                precip_data=None, discount_rate=0.03, n_years=10):
    """
    ROI_resilience = (sum(L_baseline - L_intervention) * gamma^t) / Cost +/- U_precip

    Loss proxy: risk_score * GDP * normalized_population
    """
    loss_baseline = baseline_risk * gdp_flat * (pop_flat / pop_flat.max())
    loss_intervention = intervention_risk * gdp_flat * (pop_flat / pop_flat.max())

    total_loss_avoided = 0
    for t in range(n_years):
        gamma = (1 / (1 + discount_rate)) ** t
        total_loss_avoided += np.sum(loss_baseline - loss_intervention) * gamma

    roi = total_loss_avoided / cost

    u_precip = compute_uncertainty_penalty(precip_data) if precip_data is not None else 0

    return {
        'roi': float(roi),
        'roi_lower': float(roi - u_precip),
        'roi_upper': float(roi + u_precip),
        'total_loss_avoided': float(total_loss_avoided),
        'u_precip': float(u_precip),
        'mean_risk_reduction': float(np.mean(baseline_risk - intervention_risk)),
    }


def compute_uncertainty_penalty(precip_data, ci=0.95):
    """
    U_precip: uncertainty from precipitation variability.
    Higher precip variance -> wider confidence interval -> larger penalty.
    """
    if precip_data is None:
        return 0.0

    if precip_data.ndim == 3:
        spatial_std = np.std(precip_data, axis=0)
        mean_uncertainty = np.mean(spatial_std)
    else:
        mean_uncertainty = np.std(precip_data)

    mean_precip = np.mean(np.abs(precip_data))
    u_precip = (mean_uncertainty / (mean_precip + 1e-8)) * 0.5

    return float(u_precip)
