"""
Köppen-Geiger Climate Classification Engine (Corrected).

Stabilizes GNN risk scores by providing categorical climate priors.
Handles Kelvin -> Celsius and daily flux -> monthly/annual total precipitation.
"""
import os
import platform
import multiprocessing as mp
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
    """
    Returns (T, nlat, nlon) numerical KG codes.

    Performance:
      - Default behavior is single-process to avoid "cooking" laptops.
      - On Linux clusters you can opt into CPU parallelism by setting:
          SENTREE_KG_WORKERS=<n>
        (recommended: match Slurm `--cpus-per-task`).

    Note: Parallel mode is Linux-only by default (macOS/Windows use `spawn` and
    can copy huge arrays, which is both slow and resource-heavy).
    """
    T, M, nlat, nlon = tas_monthly.shape
    results = np.zeros((T, nlat, nlon), dtype=np.int32)

    workers = int(os.environ.get("SENTREE_KG_WORKERS", "1") or "1")
    system = platform.system().lower()
    allow_parallel = (workers > 1) and (system == "linux")

    if not allow_parallel:
        for t in range(T):
            for i in range(nlat):
                for j in range(nlon):
                    _, val = classify_koppen_geiger(
                        tas_monthly[t, :, i, j],
                        pr_monthly[t, :, i, j],
                        is_kelvin,
                        is_daily_flux,
                    )
                    results[t, i, j] = val
        return results

    # --- Parallel path (Linux only) ---
    # Use fork so workers share the large numpy arrays without pickling copies.
    ctx = mp.get_context("fork")

    global _KG_TAS, _KG_PR, _KG_IS_KELVIN, _KG_IS_DAILY_FLUX
    _KG_TAS = tas_monthly
    _KG_PR = pr_monthly
    _KG_IS_KELVIN = bool(is_kelvin)
    _KG_IS_DAILY_FLUX = bool(is_daily_flux)

    def _init_worker(tas_ref, pr_ref, is_kelvin_ref, is_daily_flux_ref):
        global _KG_TAS, _KG_PR, _KG_IS_KELVIN, _KG_IS_DAILY_FLUX
        _KG_TAS = tas_ref
        _KG_PR = pr_ref
        _KG_IS_KELVIN = is_kelvin_ref
        _KG_IS_DAILY_FLUX = is_daily_flux_ref

    def _classify_lat_band(task):
        t, i0, i1 = task
        out = np.zeros((i1 - i0, nlon), dtype=np.int32)
        for ii, i in enumerate(range(i0, i1)):
            for j in range(nlon):
                _, val = classify_koppen_geiger(
                    _KG_TAS[t, :, i, j],
                    _KG_PR[t, :, i, j],
                    _KG_IS_KELVIN,
                    _KG_IS_DAILY_FLUX,
                )
                out[ii, j] = val
        return t, i0, out

    # Choose bands to keep task overhead reasonable.
    band_rows = max(1, int(np.ceil(nlat / (workers * 6))))
    tasks = []
    for t in range(T):
        for i0 in range(0, nlat, band_rows):
            i1 = min(nlat, i0 + band_rows)
            tasks.append((t, i0, i1))

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(tas_monthly, pr_monthly, _KG_IS_KELVIN, _KG_IS_DAILY_FLUX),
    ) as pool:
        for t, i0, band in pool.imap_unordered(_classify_lat_band, tasks, chunksize=1):
            results[t, i0 : i0 + band.shape[0], :] = band

    return results
