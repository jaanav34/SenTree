"""
Köppen-Geiger Climate Classification Engine (Corrected).

Stabilizes GNN risk scores by providing categorical climate priors.
Handles Kelvin -> Celsius and daily flux -> monthly/annual total precipitation.
"""
import numpy as np

# Numerical mapping for the classifications (0-31)
KG_MAP = {
    'Af': 1, 'Am': 2, 'Aw': 3, 'As': 4,
    'BWh': 5, 'BWk': 6, 'BSh': 7, 'BSk': 8,
    'Cfa': 9, 'Cfb': 10, 'Cfc': 11, 'Cwa': 12, 'Cwb': 13, 'Cwc': 14, 'Csa': 15, 'Csb': 16, 'Csc': 17,
    'Dfa': 18, 'Dfb': 19, 'Dfc': 20, 'Dfd': 21, 'Dwa': 22, 'Dwb': 23, 'Dwc': 24, 'Dwd': 25, 'Dsa': 26, 'Dsb': 27, 'Dsc': 28, 'Dsd': 29,
    'ET': 30, 'EF': 31,
    'Unknown': 0
}

KG_LABELS = {v: k for k, v in KG_MAP.items()}
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

def classify_koppen_geiger(temp_monthly, precip_monthly, is_kelvin=True, is_daily_flux=True):
    """
    Classify a grid cell using standard KG rules with proper unit scaling.
    
    Args:
        temp_monthly: (12,) Monthly temperatures.
        precip_monthly: (12,) Monthly precipitation values.
        is_kelvin: If True, converts K -> C.
        is_daily_flux: If True, scales mm/day or kg/m2/s to monthly totals.
    """
    # 1. Unit Conversion
    t = np.array(temp_monthly)
    if is_kelvin and np.mean(t) > 200:
        t = t - 273.15
        
    p = np.array(precip_monthly)
    if is_daily_flux:
        # Scale daily mean (mm/day) to monthly totals (mm)
        p = p * DAYS_IN_MONTH
        
    T_ann = np.mean(t)
    P_ann = np.sum(p)
    T_max = np.max(t)
    T_min = np.min(t)
    
    # 2. Determine Seasonality for Group B threshold (P_thresh)
    # Summer: Apr-Sep (NH)
    P_summer = np.sum(p[3:9])
    P_winter = P_ann - P_summer
    
    if P_summer / (P_ann + 1e-8) >= 0.7:
        P_thresh = 2 * T_ann + 28
    elif P_winter / (P_ann + 1e-8) >= 0.7:
        P_thresh = 2 * T_ann
    else:
        P_thresh = 2 * T_ann + 14
        
    # --- Group B (Dry) ---
    if P_ann < 10 * P_thresh:
        group = 'BW' if P_ann < 5 * P_thresh else 'BS'
        code = group + ('h' if T_ann >= 18 else 'k')
        return code, KG_MAP.get(code, 0)

    # --- Group E (Polar) ---
    if T_max < 10:
        code = 'EF' if T_max <= 0 else 'ET'
        return code, KG_MAP.get(code, 0)

    # --- Group A (Tropical) ---
    if T_min >= 18:
        P_min = np.min(p)
        if P_min >= 60:
            code = 'Af'
        else:
            if P_min >= 100 - (P_ann / 25):
                code = 'Am'
            else:
                code = 'Aw' if P_summer < P_winter else 'As'
        return code, KG_MAP.get(code, 0)

    # --- Group C (Temperate) & D (Continental) ---
    group = 'C' if T_min > 0 else 'D'
    
    # Sub-types
    P_s_min, P_s_max = np.min(p[3:9]), np.max(p[3:9])
    P_w_min, P_w_max = np.min(p[[0,1,2,9,10,11]]), np.max(p[[0,1,2,9,10,11]])
    
    if P_s_min < 40 and P_s_min < P_w_max / 3: precip_sub = 's'
    elif P_w_min < P_s_max / 10: precip_sub = 'w'
    else: precip_sub = 'f'
        
    months_above_10 = np.sum(t > 10)
    if T_max >= 22: temp_sub = 'a'
    elif months_above_10 >= 4: temp_sub = 'b'
    elif group == 'D' and T_min < -38: temp_sub = 'd'
    else: temp_sub = 'c'
        
    code = group + precip_sub + temp_sub
    return code, KG_MAP.get(code, 0)

def classify_grid(tas_monthly, pr_monthly, is_kelvin=True, is_daily_flux=True):
    """Returns (T, nlat, nlon) numerical KG codes."""
    T, M, nlat, nlon = tas_monthly.shape
    results = np.zeros((T, nlat, nlon), dtype=np.int32)
    for t in range(T):
        for i in range(nlat):
            for j in range(nlon):
                _, val = classify_koppen_geiger(tas_monthly[t, :, i, j], pr_monthly[t, :, i, j], 
                                                is_kelvin, is_daily_flux)
                results[t, i, j] = val
    return results
