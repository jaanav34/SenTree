"""
Köppen-Geiger Climate Classification Engine.

Implements the standard Köppen-Geiger classification rules:
  - Group A: Tropical
  - Group B: Dry
  - Group C: Temperate
  - Group D: Continental
  - Group E: Polar

Based on the Wikipedia specification: https://en.wikipedia.org/wiki/Köppen_climate_classification
"""
import numpy as np

# Numerical mapping for the classifications
KG_MAP = {
    'Af': 1, 'Am': 2, 'Aw': 3, 'As': 4,
    'BWh': 5, 'BWk': 6, 'BSh': 7, 'BSk': 8,
    'Cfa': 9, 'Cfb': 10, 'Cfc': 11, 'Cwa': 12, 'Cwb': 13, 'Cwc': 14, 'Csa': 15, 'Csb': 16, 'Csc': 17,
    'Dfa': 18, 'Dfb': 19, 'Dfc': 20, 'Dfd': 21, 'Dwa': 22, 'Dwb': 23, 'Dwc': 24, 'Dwd': 25, 'Dsa': 26, 'Dsb': 27, 'Dsc': 28, 'Dsd': 29,
    'ET': 30, 'EF': 31,
    'Unknown': 0
}

KG_LABELS = {v: k for k, v in KG_MAP.items()}

def classify_koppen_geiger(temp_monthly, precip_monthly):
    """
    Classify a single grid cell into a Köppen-Geiger climate zone.
    
    Args:
        temp_monthly: (12,) Monthly average temperature in °C.
        precip_monthly: (12,) Monthly average precipitation in mm.
        
    Returns:
        KG_Code: string (e.g., 'Af', 'Cfa')
        KG_Value: int (numerical index for mapping)
    """
    T_ann = np.mean(temp_monthly)
    P_ann = np.sum(precip_monthly)
    
    T_max = np.max(temp_monthly)
    T_min = np.min(temp_monthly)
    
    # Determine seasonality for Group B
    # 70% or more of P_ann falls in summer half of year
    # Summer: Apr-Sep (NH) or Oct-Mar (SH). For simplification, we check both.
    P_summer_nh = np.sum(precip_monthly[3:9])
    P_summer_sh = np.sum(precip_monthly[[9, 10, 11, 0, 1, 2]])
    
    if P_summer_nh / (P_ann + 1e-8) >= 0.7:
        P_thresh = 2 * T_ann + 28
    elif P_summer_sh / (P_ann + 1e-8) >= 0.7:
        P_thresh = 2 * T_ann + 28
    elif np.sum(precip_monthly[[0,1,2,9,10,11]]) / (P_ann + 1e-8) >= 0.7: # Winter-wet
        P_thresh = 2 * T_ann
    else:
        P_thresh = 2 * T_ann + 14
        
    # --- Group B (Dry) ---
    if P_ann < 10 * P_thresh:
        if P_ann < 5 * P_thresh:
            group = 'BW'
        else:
            group = 'BS'
        
        if T_ann >= 18:
            code = group + 'h'
        else:
            code = group + 'k'
        return code, KG_MAP.get(code, 0)

    # --- Group E (Polar) ---
    if T_max < 10:
        if T_max > 0:
            code = 'ET'
        else:
            code = 'EF'
        return code, KG_MAP.get(code, 0)

    # --- Group A (Tropical) ---
    if T_min >= 18:
        P_min = np.min(precip_monthly)
        if P_min >= 60:
            code = 'Af'
        else:
            # Check for Monsoon or Savanna
            if P_min >= 100 - (P_ann / 25):
                code = 'Am'
            else:
                # Aw or As (Winter dry vs Summer dry)
                if P_summer_nh < P_summer_sh: # NH Winter Dry
                    code = 'Aw'
                else:
                    code = 'As'
        return code, KG_MAP.get(code, 0)

    # --- Group C (Temperate) and D (Continental) ---
    # T_min between -3 (or 0) and 18 -> Group C
    # T_min <= -3 (or 0) -> Group D
    # We use 0°C as the threshold for D (modern standard)
    if T_min > 0:
        group = 'C'
    else:
        group = 'D'
        
    # Precipitation sub-type
    # f: no dry season
    # s: dry summer
    # w: dry winter
    P_s_min = np.min(precip_monthly[3:9]) # Summer (NH proxy)
    P_s_max = np.max(precip_monthly[3:9])
    P_w_min = np.min(precip_monthly[[9, 10, 11, 0, 1, 2]]) # Winter (NH proxy)
    P_w_max = np.max(precip_monthly[[9, 10, 11, 0, 1, 2]])
    
    if P_s_min < 40 and P_s_min < P_w_max / 3:
        precip_sub = 's'
    elif P_w_min < P_s_max / 10:
        precip_sub = 'w'
    else:
        precip_sub = 'f'
        
    # Temperature sub-type
    # a: hot summer (T_max >= 22)
    # b: warm summer (T_max < 22, at least 4 months > 10)
    # c: cool summer (1-3 months > 10)
    # d: extremely continental (Group D only, T_min < -38)
    months_above_10 = np.sum(temp_monthly > 10)
    
    if T_max >= 22:
        temp_sub = 'a'
    elif months_above_10 >= 4:
        temp_sub = 'b'
    elif group == 'D' and T_min < -38:
        temp_sub = 'd'
    else:
        temp_sub = 'c'
        
    code = group + precip_sub + temp_sub
    return code, KG_MAP.get(code, 0)

def classify_grid(temp_3d, precip_3d):
    """
    Classify an entire grid over a multi-year period (returning classification per year).
    
    Args:
        temp_3d: (T, 12, nlat, nlon) monthly data
        precip_3d: (T, 12, nlat, nlon) monthly data
        
    Returns:
        kg_codes: (T, nlat, nlon) numerical KG values
    """
    T, _, nlat, nlon = temp_3d.shape
    kg_results = np.zeros((T, nlat, nlon), dtype=np.int32)
    
    for t in range(T):
        for i in range(nlat):
            for j in range(nlon):
                _, val = classify_koppen_geiger(temp_3d[t, :, i, j], precip_3d[t, :, i, j])
                kg_results[t, i, j] = val
                
    return kg_results
