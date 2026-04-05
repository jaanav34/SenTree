"""SenTree Dashboard â€” Streamlit UI."""
import os
import sys
import json
from pathlib import Path
from textwrap import dedent

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from sentree_venv import ensure_venv

ensure_venv()

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection

from src.simulation.interventions import INTERVENTIONS

st.set_page_config(page_title='SenTree - Resilience ROI Dashboard', layout='wide', initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --sentree-ink: #18322d;
        --sentree-ink-soft: #29443d;
        --sentree-ink-muted: #5a6f69;
        --sentree-accent: #0f766e;
        --sentree-accent-warm: #b45309;
        --sentree-card-top: rgba(255, 251, 245, 0.96);
        --sentree-card-bottom: rgba(245, 249, 246, 0.94);
        --sentree-sidebar-bg: linear-gradient(180deg, #17342f 0%, #1f2d3d 100%);
        --sentree-sidebar-ink: #f4eedf;
        --sentree-sidebar-muted: #d8d1bf;
        --sentree-sidebar-field: rgba(251, 247, 238, 0.96);
        --sentree-sidebar-border: rgba(244, 238, 223, 0.18);
    }

    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }

    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 30%),
            radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 28%),
            linear-gradient(180deg, #f1eee2 0%, #e4efe8 54%, #f6f3ea 100%);
        color: var(--sentree-ink);
        overflow-x: hidden;
    }

    html, body {
        overflow-x: hidden;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .sentree-hero {
        position: relative;
        overflow: hidden;
        padding: 1.6rem 1.7rem 1.5rem 1.7rem;
        border-radius: 28px;
        background:
            radial-gradient(circle at 85% 18%, rgba(255, 255, 255, 0.34), transparent 20%),
            radial-gradient(circle at 10% 5%, rgba(15, 118, 110, 0.16), transparent 22%),
            linear-gradient(135deg, rgba(255, 252, 246, 0.95), rgba(232, 243, 237, 0.92));
        border: 1px solid rgba(23, 52, 47, 0.12);
        box-shadow: 0 20px 50px rgba(23, 52, 47, 0.10);
        margin-bottom: 1.4rem;
    }

    .sentree-hero::after {
        content: "";
        position: absolute;
        inset: auto -5% -40% auto;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(180, 83, 9, 0.12), transparent 68%);
        pointer-events: none;
    }

    .sentree-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(23, 52, 47, 0.08);
        color: var(--sentree-accent);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sentree-hero h1 {
        margin: 0.85rem 0 0.4rem 0;
        font-size: 3rem;
        line-height: 0.96;
        max-width: 8.5ch;
    }

    .sentree-hero p {
        margin: 0;
        max-width: 54rem;
        color: var(--sentree-ink-soft);
        font-size: 1rem;
    }

    .sentree-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.1rem;
    }

    .sentree-badge {
        padding: 0.48rem 0.72rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(23, 52, 47, 0.10);
        color: #17342f;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .sentree-section {
        margin: 1.35rem 0 0.35rem 0;
    }

    .sentree-section-label {
        color: var(--sentree-accent);
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .sentree-section h2 {
        margin: 0;
        font-size: 2rem;
    }

    .sentree-section p {
        margin: 0.28rem 0 0 0;
        color: var(--sentree-ink-soft);
        max-width: 55rem;
    }

    .sentree-card {
        padding: 1.1rem 1.15rem;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255, 252, 246, 0.94), rgba(242, 248, 244, 0.92));
        border: 1px solid rgba(23, 52, 47, 0.10);
        box-shadow: 0 14px 38px rgba(23, 52, 47, 0.08);
        margin-bottom: 1rem;
    }

    .sentree-card h3 {
        margin: 0;
        font-size: 1.08rem;
    }

    .sentree-card p {
        margin: 0.38rem 0 0 0;
        color: var(--sentree-ink-soft);
        font-size: 0.95rem;
    }

    .sentree-kpi {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(255, 250, 241, 0.96), rgba(236, 245, 239, 0.93));
        border: 1px solid rgba(23, 52, 47, 0.10);
        box-shadow: 0 12px 32px rgba(23, 52, 47, 0.08);
        min-height: 118px;
    }

    .sentree-kpi-label {
        color: var(--sentree-ink-muted);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .sentree-kpi-value {
        margin-top: 0.48rem;
        color: #17342f;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }

    .sentree-kpi-sub {
        margin-top: 0.42rem;
        color: var(--sentree-ink-soft);
        font-size: 0.92rem;
    }

    h1, h2, h3 {
        color: #17342f;
        letter-spacing: -0.02em;
    }

    section[data-testid="stSidebar"] > div {
        background: var(--sentree-sidebar-bg);
        border-right: 1px solid rgba(244, 238, 223, 0.12);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: var(--sentree-sidebar-ink) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: var(--sentree-sidebar-ink) !important;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, var(--sentree-card-top), var(--sentree-card-bottom));
        border: 1px solid rgba(23, 52, 47, 0.16);
        border-radius: 18px;
        padding: 0.85rem 1rem;
        box-shadow: 0 14px 40px rgba(23, 52, 47, 0.10);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--sentree-ink-muted) !important;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: #17342f !important;
        font-weight: 800;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--sentree-accent) !important;
        font-weight: 700;
    }

    div.stButton > button, div[data-testid="stFormSubmitButton"] button {
        border-radius: 999px;
        border: 1px solid rgba(23, 52, 47, 0.22);
        background: #17342f;
        color: #f8f4ea;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(23, 52, 47, 0.16);
    }

    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--sentree-accent);
        border-color: var(--sentree-accent);
        color: #fffdf7;
    }

    div.stButton > button p, div[data-testid="stFormSubmitButton"] button p {
        color: inherit !important;
        font-weight: 700;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background: rgba(255, 251, 245, 0.92);
        border-color: rgba(23, 52, 47, 0.20);
        color: #17342f;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="base-input"] input {
        color: #17342f !important;
    }

    div[data-testid="stWidgetLabel"] *,
    label[data-baseweb="checkbox"] span,
    .stSelectbox label,
    .stSlider label,
    .stTextInput label {
        color: #23423c !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
        background: var(--sentree-sidebar-field);
        border-color: var(--sentree-sidebar-border);
        color: #18322d;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="base-input"] input {
        color: #18322d !important;
    }

    section[data-testid="stSidebar"] .sentree-card {
        background: linear-gradient(180deg, rgba(255, 251, 245, 0.98), rgba(241, 247, 244, 0.95));
        border-color: rgba(23, 52, 47, 0.14);
        box-shadow: 0 12px 28px rgba(8, 18, 22, 0.18);
    }

    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] .sentree-card h3 {
        color: #17342f !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] .sentree-card p {
        color: #49615a !important;
    }

    section[data-testid="stSidebar"] div.stButton > button {
        background: rgba(244, 238, 223, 0.12);
        border-color: rgba(244, 238, 223, 0.22);
        color: var(--sentree-sidebar-ink);
        box-shadow: none;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(15, 118, 110, 0.35);
        border-color: rgba(15, 118, 110, 0.45);
    }

    section[data-testid="stSidebar"] [data-baseweb="checkbox"] > div {
        color: var(--sentree-sidebar-ink);
    }

    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"] {
        color: #42615a;
    }

    div[data-baseweb="slider"] [role="slider"] {
        background: var(--sentree-accent);
        border-color: var(--sentree-accent);
    }

    div[data-baseweb="slider"] > div > div {
        background: rgba(15, 118, 110, 0.22);
    }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCaptionContainer"] {
        color: var(--sentree-ink-soft);
    }

    div[data-testid="stAlert"] {
        border-radius: 16px;
    }

    div[data-testid="stExpander"] {
        border-radius: 18px;
        border: 1px solid rgba(23, 52, 47, 0.10);
        background: rgba(255, 252, 246, 0.8);
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        background: rgba(255, 252, 246, 0.84);
    }

    button[kind="tab"] {
        border-radius: 999px;
        border: 1px solid rgba(23, 52, 47, 0.10);
        background: rgba(255, 252, 246, 0.72);
        padding: 0.4rem 0.95rem;
    }

    button[kind="tab"][aria-selected="true"] {
        background: #17342f;
        color: #f8f4ea;
    }

    hr {
        border-color: rgba(23, 52, 47, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
_WORLD_LON_SPAN_DEG = 360.0
_WORLD_LAT_SPAN_DEG = 180.0
_WEB_MERCATOR_MAX_LAT = 85.05112878


def _show_video(path_or_url: str) -> None:
    """Render a video from a local path (cluster filesystem) or a URL."""
    p = Path(path_or_url)
    if p.exists():
        st.video(p.read_bytes(), format="video/mp4")
        return
    st.video(path_or_url)


def section_header(label: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="sentree-section">
            <div class="sentree-section-label">{label}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def surface_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="sentree-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="sentree-kpi">
            <div class="sentree-kpi-label">{label}</div>
            <div class="sentree-kpi-value">{value}</div>
            <div class="sentree-kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Runtime control state (rendered later near AI summary) ---
scenario = "SSP3-7.0 (High Emissions)"
intervention = "Investor Mode"
capital_allocation_m = int(st.session_state.get("capital_allocation_m", 50))
capital_allocation = int(capital_allocation_m) * 1_000_000

# --- Load results ---
@st.cache_data
def load_roi_data():
    roi_path = 'outputs/roi/roi_results.json'
    if os.path.exists(roi_path):
        with open(roi_path, 'r') as f:
            return json.load(f)
    return {
        'mangrove_restoration': {
            'name': 'Coastal Mangrove Restoration',
            'roi': 3.42, 'roi_lower': 2.91, 'roi_upper': 3.93,
            'total_loss_avoided': 3_420_000_000,
            'mean_risk_reduction': 0.18,
            'tail_risk_nodes_neutralized': 12,
        },
        'regenerative_agriculture': {
            'name': 'Regenerative Agriculture',
            'roi': 2.15, 'roi_lower': 1.78, 'roi_upper': 2.52,
            'total_loss_avoided': 2_150_000_000,
            'mean_risk_reduction': 0.09,
            'tail_risk_nodes_neutralized': 5,
        }
    }


@st.cache_data
def load_risk_timeseries():
    path = "outputs/roi/risk_timeseries.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_training_history():
    path = "outputs/roi/gnn_training_history.npz"
    if not os.path.exists(path):
        return None

    with np.load(path) as data:
        history = {key: data[key] for key in data.files}

    predictions = history["predictions"]
    history["epochs"] = np.arange(1, predictions.shape[0] + 1, dtype=np.int32)
    history["mean_risk"] = predictions.mean(axis=1)
    history["p95_risk"] = np.percentile(predictions, 95, axis=1)
    history["max_risk"] = predictions.max(axis=1)
    history["tail_threshold"] = float(np.percentile(history["target"], 95))
    return history


@st.cache_data
def load_opportunity_map():
    path = "outputs/roi/opportunity_map.npz"
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    return {
        "total_reduction_map": data["total_reduction_map"],
        "tail_flags": data["tail_flags"].astype(bool),
        "lats": data["lats"],
        "lons": data["lons"],
        "years": data["years"],
    }


roi_data = load_roi_data()


def _format_money_short(value):
    value = float(value)
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"${value/1e9:.2f}B"
    if abs_val >= 1e6:
        return f"${value/1e6:.1f}M"
    if abs_val >= 1e3:
        return f"${value/1e3:.1f}K"
    return f"${value:,.0f}"


def _apply_capital_allocation(roi_data, capital_allocation):
    """Return ROI data adjusted for the capital allocation slider."""
    adjusted = {}
    for key, data in roi_data.items():
        entry = dict(data)
        base_cost = INTERVENTIONS.get(key, {}).get("cost_usd")
        if not base_cost:
            adjusted[key] = entry
            continue

        base_cost = float(base_cost)
        ratio = float(capital_allocation) / base_cost
        # Keep small-budget ROI from inflating unrealistically:
        # - For ratio <= 1, use linear scaling (ROI stays near baseline).
        # - For ratio > 1, apply sublinear scaling (diminishing returns).
        if ratio <= 1.0:
            impact_scale = ratio
        else:
            impact_scale = ratio ** 0.85

        base_roi = float(entry.get("roi", 0.0))
        base_loss = float(entry.get("total_loss_avoided", 0.0))
        adjusted_loss = base_loss * impact_scale
        adjusted_roi = (base_roi * base_cost * impact_scale) / float(capital_allocation)

        entry["roi"] = adjusted_roi
        entry["total_loss_avoided"] = adjusted_loss
        entry["capital_allocation"] = float(capital_allocation)
        entry["impact_scale"] = impact_scale
        adjusted[key] = entry
    return adjusted


roi_data_adjusted = _apply_capital_allocation(roi_data, capital_allocation)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km)."""
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def _wrap_lon_180(lon):
    """Normalize longitude to [-180, 180] for WebMercator map renderers (pydeck/mapbox)."""
    lon = np.asarray(lon, dtype=np.float64)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lon = np.where(lon < -180.0, lon + 360.0, lon)
    return lon


def _suggest_pydeck_zoom(lat_span_deg: float, lon_span_deg: float) -> float:
    """Heuristic zoom so the bbox roughly fits (works for SE Asia and global)."""
    lat_span_deg = float(lat_span_deg) if lat_span_deg is not None else 0.0
    lon_span_deg = float(lon_span_deg) if lon_span_deg is not None else 0.0

    lat_span_deg = max(lat_span_deg, 1e-6)
    lon_span_deg = max(lon_span_deg, 1e-6)

    zoom_lon = np.log2(_WORLD_LON_SPAN_DEG / lon_span_deg) + 1.0
    zoom_lat = np.log2(_WORLD_LAT_SPAN_DEG / lat_span_deg) + 1.0
    zoom = float(min(zoom_lon, zoom_lat))
    return float(np.clip(zoom, 0.5, 6.0))


def _approx_cell_size_m(lats: np.ndarray, lons: np.ndarray) -> int:
    """Approximate meters per grid step (used for pydeck cell sizing)."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if lats.size < 2 or lons.size < 2:
        return 55_000

    dlat = np.nanmedian(np.abs(np.diff(lats)))
    dlon = np.nanmedian(np.abs(np.diff(lons)))
    step_deg = float(np.nanmax([dlat, dlon]))

    # deck.gl layers size cells in meters in WebMercator space. Longitudinal spacing
    # in meters collapses near the poles by ~cos(latitude). Using an equator-based
    # meter size causes high-latitude rows to merge into bands; scaling by max-lat
    # makes cells too tiny to see when zoomed out.
    #
    # Use a mid-latitude reference and clamp so global views remain visible.
    abs_lats = np.abs(lats)
    ref_lat = float(np.nanpercentile(abs_lats, 60))  # ~mid-high latitude
    cos_factor = float(np.cos(np.deg2rad(ref_lat)))
    cos_factor = float(np.clip(cos_factor, 0.25, 1.0))

    # 1Â° latitude ~= 111 km (good enough for UI sizing)
    cell = int(round(111_000.0 * step_deg * cos_factor))
    return int(np.clip(cell, 20_000, 350_000))


@st.cache_data
def _opportunity_points(opportunity):
    """Flatten grids into a dataframe with nearest-city labels (offline)."""
    lats = opportunity["lats"]
    lons = opportunity["lons"]
    value_grid = opportunity["total_reduction_map"]
    flags_grid = opportunity["tail_flags"]

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_flat = lat_grid.flatten()
    lon_flat = _wrap_lon_180(lon_grid.flatten())
    values = value_grid.flatten()
    flags = flags_grid.flatten()

    # Minimal offline "real-life location" labeling: nearest major city.
    # (Avoids external reverse-geocoding calls.)
    cities = [
        ("Bangkok", 13.7563, 100.5018),
        ("Jakarta", -6.2088, 106.8456),
        ("Manila", 14.5995, 120.9842),
        ("Singapore", 1.3521, 103.8198),
        ("Kuala Lumpur", 3.1390, 101.6869),
        ("Ho Chi Minh City", 10.8231, 106.6297),
        ("Hanoi", 21.0278, 105.8342),
        ("Phnom Penh", 11.5564, 104.9282),
        ("Vientiane", 17.9757, 102.6331),
        ("Yangon", 16.8409, 96.1735),
        ("Cebu", 10.3157, 123.8854),
        ("Davao", 7.1907, 125.4553),
        ("Denpasar", -8.6705, 115.2126),
        ("Surabaya", -7.2575, 112.7521),
        ("Medan", 3.5952, 98.6722),
    ]

    city_names = np.array([c[0] for c in cities], dtype=object)
    city_lats = np.array([c[1] for c in cities], dtype=np.float64)
    city_lons = np.array([c[2] for c in cities], dtype=np.float64)

    # Compute nearest city for each cell (vectorized over cities)
    # Dist matrix: (n_points, n_cities)
    dists = np.stack([
        _haversine_km(lat_flat, lon_flat, city_lats[i], city_lons[i]) for i in range(len(cities))
    ], axis=1)
    idx = np.argmin(dists, axis=1)
    nearest = city_names[idx]
    nearest_km = dists[np.arange(dists.shape[0]), idx]

    # Normalize values for color mapping
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
    norm = np.clip((values - vmin) / denom, 0, 1)

    # Green ramp (light -> deep)
    colors = np.column_stack([
        (40 * (1 - norm)).astype(int),          # R (slight tint)
        (220 * norm + 30).astype(int),          # G
        (40 * (1 - norm)).astype(int),          # B (slight tint)
        np.full_like(norm, 160, dtype=int),     # A
    ])

    df = pd.DataFrame({
        "lat": lat_flat.astype(float),
        "lon": lon_flat.astype(float),
        "value": values.astype(float),
        "tail_flag": flags.astype(bool),
        "nearest_city": nearest,
        "nearest_km": nearest_km.astype(float),
    })
    df["color"] = colors.tolist()

    # WebMercator (deck.gl basemaps) only supports latitudes up to about Â±85.051Â°.
    # Global ISIMIP grids often include points up to ~Â±89Â°, which will render "off-map"
    # in the void. Filter those points for the interactive basemap view.
    df = df[np.abs(df["lat"]) <= _WEB_MERCATOR_MAX_LAT].reset_index(drop=True)
    return df, (vmin, vmax)


def build_training_figure(training, epoch_idx, show_edges=True, highlight_targets=True):
    positions = training["positions"]
    lats = positions[:, 0]
    lons = positions[:, 1]
    pred = training["predictions"][epoch_idx]
    loss = training["loss"]
    mean_risk = training["mean_risk"]
    p95_risk = training["p95_risk"]
    target = training["target"]

    fig = plt.figure(figsize=(14, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1], wspace=0.22, hspace=0.3)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_risk = fig.add_subplot(grid[1, 1])

    if show_edges and training["edge_index"].size > 0:
        edge_index = training["edge_index"]
        segments = [
            [(lons[src], lats[src]), (lons[dst], lats[dst])]
            for src, dst in zip(edge_index[0], edge_index[1])
        ]
        edge_collection = LineCollection(segments, colors=(0.09, 0.2, 0.18, 0.08), linewidths=0.5)
        ax_map.add_collection(edge_collection)

    scatter = ax_map.scatter(
        lons,
        lats,
        c=pred,
        s=14 + pred * 26,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
        linewidths=0,
    )

    if highlight_targets:
        tail_mask = target >= training["tail_threshold"]
        ax_map.scatter(
            lons[tail_mask],
            lats[tail_mask],
            s=44,
            facecolors="none",
            edgecolors="#17342f",
            linewidths=0.9,
            alpha=0.75,
        )

    ax_map.set_title(f"Node Risk Field at Epoch {epoch_idx + 1}", loc="left", fontsize=13, fontweight="bold")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_facecolor("#fffdf7")
    ax_map.grid(alpha=0.12)

    cbar = fig.colorbar(scatter, ax=ax_map, fraction=0.035, pad=0.02)
    cbar.set_label("Predicted systemic risk")

    epochs = training["epochs"]
    ax_loss.plot(epochs, loss, color="#0f766e", linewidth=2.2)
    ax_loss.scatter([epoch_idx + 1], [loss[epoch_idx]], color="#ea580c", s=52, zorder=3)
    ax_loss.axvline(epoch_idx + 1, color="#ea580c", linestyle="--", linewidth=1.1, alpha=0.7)
    ax_loss.set_title("Optimization Progress", loc="left", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Huber loss")
    ax_loss.grid(alpha=0.18)
    ax_loss.set_facecolor("#fffdf7")

    ax_risk.plot(epochs, mean_risk, color="#2563eb", linewidth=2.2, label="Mean risk")
    ax_risk.plot(epochs, p95_risk, color="#b91c1c", linewidth=2.2, label="95th pct risk")
    ax_risk.scatter([epoch_idx + 1], [mean_risk[epoch_idx]], color="#2563eb", s=48, zorder=3)
    ax_risk.scatter([epoch_idx + 1], [p95_risk[epoch_idx]], color="#b91c1c", s=48, zorder=3)
    ax_risk.axvline(epoch_idx + 1, color="#ea580c", linestyle="--", linewidth=1.1, alpha=0.7)
    ax_risk.set_title("Prediction Profile", loc="left", fontsize=12, fontweight="bold")
    ax_risk.set_xlabel("Epoch")
    ax_risk.set_ylabel("Risk score")
    ax_risk.grid(alpha=0.18)
    ax_risk.set_facecolor("#fffdf7")
    ax_risk.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.patch.set_facecolor("#fffaf2")
    return fig


def render_training_frame(viz_placeholder, metrics_placeholder, training, epoch_idx, show_edges=True, highlight_targets=True):
    pred = training["predictions"][epoch_idx]
    prev = training["predictions"][epoch_idx - 1] if epoch_idx > 0 else pred
    tail_mask = training["target"] >= training["tail_threshold"]

    with metrics_placeholder.container():
        cols = st.columns(4)
        cols[0].metric("Epoch", f"{epoch_idx + 1}/{len(training['epochs'])}")
        cols[1].metric(
            "Training Loss",
            f"{training['loss'][epoch_idx]:.4f}",
            f"{(training['loss'][epoch_idx] - training['loss'][epoch_idx - 1]):+.4f}" if epoch_idx > 0 else None,
        )
        cols[2].metric(
            "Mean Node Risk",
            f"{pred.mean():.3f}",
            f"{(pred.mean() - prev.mean()):+.3f}" if epoch_idx > 0 else None,
        )
        cols[3].metric(
            "Neutralized Tail Nodes",
            f"{int((tail_mask & (pred < training['tail_threshold'])).sum())}",
            f"Tracked: {int(tail_mask.sum())}",
        )

    fig = build_training_figure(training, epoch_idx, show_edges=show_edges, highlight_targets=highlight_targets)
    viz_placeholder.pyplot(fig, use_container_width=True)
    plt.close(fig)


def build_risk_timeseries_figure(ts, metric, intervention_key=None, intervention_name=None):
    years = np.array(ts["years"])
    fig, ax = plt.subplots(figsize=(12.8, 4.6))

    series = [("Baseline", ts["baseline"][metric], "#17342f")]
    if intervention_key and intervention_key in ts:
        label = intervention_name or intervention_key.replace("_", " ").title()
        series.append((label, ts[intervention_key][metric], "#0f766e"))

    for name, values, color in series:
        values_arr = np.array(values)
        ax.plot(years, values_arr, linewidth=2.4, color=color, label=name)
        ax.scatter([years[-1]], [values_arr[-1]], color=color, s=42, zorder=3)

    ax.set_title("Systemic Risk Trajectory", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("P95 RISK" if metric == "p95" else metric.upper())
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, loc="upper left", ncols=len(series))
    ax.set_facecolor("#fffdf7")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.patch.set_facecolor("#fffaf2")
    return fig


def render_math_view() -> None:
    section_header(
        "Explain",
        "Technical deep-dive",
        "Inspect the mathematical assumptions behind tail-risk escalation, ROI estimation, and the graph model itself.",
    )

    st.subheader("1. Tail-Risk Escalation")
    st.markdown(
        dedent(
            """
            The tail-risk engine identifies nodes where climate volatility and momentum intersect to create regime shifts.

            **A. EWMA smoothing**
            Raw climate signals are smoothed to suppress short-lived noise before momentum is measured.
            """
        )
    )
    st.latex(r"\lambda(t) = \alpha X(t) + (1-\alpha)\lambda(t-1)")
    st.markdown("where $\\alpha = 0.3$ is the decay factor.")

    st.markdown(
        dedent(
            """
            **B. Standardized momentum**
            Momentum captures how quickly the smoothed signal is accelerating relative to local volatility.
            """
        )
    )
    st.latex(r"m(t) = \frac{\lambda(t)-\lambda(t-1)}{\sigma_w(t)+\epsilon}")

    st.markdown(
        dedent(
            """
            **C. Rolling volatility**
            Volatility measures the stability of the momentum process over a local window.
            """
        )
    )
    st.latex(r"v(t) = \sqrt{\frac{1}{w}\sum_{i=t-w+1}^{t}[m(i)-\bar{m}_w]^2}")

    st.markdown(
        dedent(
            """
            **D. Hawkes self-excitation**
            A Hawkes-style intensity term helps the system capture clustered extreme events rather than isolated spikes.
            """
        )
    )
    st.latex(r"\lambda^*(t) = \mu + \sum_{t_i < t}\beta e^{-\gamma(t-t_i)}")
    st.markdown("Nodes above the 95th percentile of the composite score are flagged as tail-risk escalation zones.")

    st.subheader("2. Resilience ROI And Economic Exposure")
    st.markdown(
        dedent(
            """
            The opportunity map represents avoided damage potential under each intervention.

            **A. Loss proxy**
            Loss is modeled as climate risk interacting with GDP, population, and a regional scaling factor.
            """
        )
    )
    st.latex(r"L_{node} = R_{score}\times(G_{norm}\times P_{norm})\times S")

    st.markdown(
        dedent(
            """
            **B. Resilience ROI**
            Returns are based on discounted avoided losses over a ten-year horizon.
            """
        )
    )
    st.latex(r"ROI_{resilience} = \frac{\sum_{t=1}^{10}(L_{baseline}-L_{intervention})\times(1+r)^{-t}}{Cost}")

    st.markdown(
        dedent(
            """
            **C. Multi-source uncertainty**
            Uncertainty is combined in quadrature to keep precipitation, model, and scenario error visible.
            """
        )
    )
    st.latex(r"U_{total} = \sqrt{U_{precip}^2 + U_{model}^2 + U_{scenario}^2}")

    st.subheader("3. GNN Risk Architecture")
    st.markdown("The GNN propagates node-level risk through geographic and economic links using graph attention.")
    st.latex(r"h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\mathbf{W}h_j^{(l)}\right)")

    st.subheader("4. Climate Regime Classification")
    st.markdown(
        dedent(r"""
            KÃ¶ppen-Geiger classes are used as climate-relative context so stabilization is measured against each node's regime.

            - Group A: tropical, where $T_{min} \ge 18^\circ C$
            - Group B: dry, where annual precipitation is below the dryness threshold
            - Group C: temperate, where $0^\circ C < T_{min} < 18^\circ C$
            - Group D: continental, where $T_{min} \le 0^\circ C$
            - Group E: polar, where $T_{max} < 10^\circ C$
            """)
    )


top_intervention = max(roi_data_adjusted.values(), key=lambda item: item.get("roi", 0.0))
training_history_path = "outputs/roi/gnn_training_history.npz"
training_status = "Training snapshots ready" if os.path.exists(training_history_path) else "Run pipeline to generate playback"
video_count = len([f for f in os.listdir('outputs/videos') if f.endswith('.mp4')]) if os.path.exists('outputs/videos') else 0

def _confidence_proxy(entry):
    u_precip = float(entry.get("u_precip", 0.0))
    u_model = float(entry.get("u_model", 0.0))
    u_scenario = float(entry.get("u_scenario", 0.0))
    total_u = min(u_precip + u_model + u_scenario, 0.95)
    return max(0.55, 1.0 - total_u)

summary_conf = _confidence_proxy(top_intervention)
summary_tail = int(top_intervention.get("tail_risk_nodes_neutralized", 0))
summary_name = top_intervention.get("name", "the selected intervention")
summary_roi = float(top_intervention.get("roi", 0.0))
summary_loss = _format_money_short(top_intervention.get("total_loss_avoided", 0.0))

st.markdown(
    f"""
    <div class="sentree-hero">
        <div class="sentree-kicker">Climate adaptation intelligence</div>
        <h1>SenTree</h1>
        <p>
            A decision cockpit for climate-risk propagation, resilience ROI, and intervention storytelling.
            Explore where the graph neural network sees cascading risk, how mitigation strategies alter the map,
            and which assets deserve immediate attention.
        </p>
        <div class="sentree-badges">
            <div class="sentree-badge">Scenario: {scenario}</div>
            <div class="sentree-badge">Focus: {intervention}</div>
            <div class="sentree-badge">Region: SE Asia Coastal</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

control_cols = st.columns([2.5, 1, 1])
with control_cols[0]:
    capital_allocation_m = st.slider(
        "Total Capital Allocation (USD)",
        min_value=5,
        max_value=100,
        value=capital_allocation_m,
        step=5,
        format="$%dM",
        key="capital_allocation_m",
        help="Investor-mode sensitivity: adjust capital to see ROI and loss avoided update with a diminishing-returns assumption.",
    )
with control_cols[1]:
    st.markdown("**Region**")
    st.caption("SE Asia Coastal")
with control_cols[2]:
    st.markdown("**Horizon**")
    st.caption("2015-2100")

st.markdown(
    f"""
    <div class="sentree-card">
        <div class="sentree-section-label">AI Resilience Summary</div>
        <p><strong>Our GNN has flagged {summary_tail} tail-risk nodes</strong> in the SE Asia coastal corridor.</p>
        <p>By allocating {_format_money_short(capital_allocation)} toward {summary_name}, the fund can avoid about
        {summary_loss} in projected GDP losses by 2045.</p>
        <p>That represents a {summary_roi:.2f}x Resilience ROI with {summary_conf*100:.0f}% confidence
        (proxy based on model uncertainty).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_cols = st.columns(3)
with hero_cols[0]:
    kpi_card("Best ROI", f"{top_intervention.get('roi', 0):.2f}x", top_intervention["name"])
with hero_cols[1]:
    kpi_card("Loss Avoided", _format_money_short(top_intervention.get('total_loss_avoided', 0)), "Top intervention impact")
with hero_cols[2]:
    kpi_card("Rendered Videos", str(video_count), "Simulation clips available")

st.caption(
    f"Investor mode: metrics scale with ${capital_allocation/1e6:.0f}M total capital "
    "(sublinear, diminishing-returns assumption)."
)

active_view = st.radio(
    "View",
    ["Dashboard", "GNN Playback", "Math"],
    horizontal=True,
    label_visibility="collapsed",
    key="sentree_view",
)

if active_view == "Dashboard":
    section_header(
        "Discover",
        "Search the simulation library",
        "Use natural language to find the most relevant intervention videos, then inspect the ROI metrics that support each result.",
    )
    search_cols = st.columns([3.2, 1])
    with search_cols[0]:
        query = st.text_input(
            'Search simulations',
            placeholder='e.g., "Show where mangroves prevent collapse"',
            key='search_query'
        )
    with search_cols[1]:
        search_btn = st.button('Search', type='primary', use_container_width=True)

    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query_text" not in st.session_state:
        st.session_state.search_query_text = ""

if active_view == "Dashboard":
    # --- Search Results ---
    if search_btn and query:
        try:
            from src.embedding.vectordb import VideoSearchDB
            db = VideoSearchDB()

            if db.count() > 0:
                st.session_state.search_results = db.query(query, n_results=3)
                st.session_state.search_query_text = query
            else:
                st.session_state.search_results = "empty"
        except Exception as e:
            st.session_state.search_results = {"error": str(e)}

    if st.session_state.search_results is not None:
        surface_card("Search Results", "Ranked video matches using vector search and intervention metadata.")
        results = st.session_state.search_results

        if results == "empty":
            st.info('No videos indexed yet. Run `python scripts/index_videos.py` first.')
        elif isinstance(results, dict) and results.get("error"):
            st.error(f"Search error: {results['error']}")
            st.info('Showing demo results instead.')
        else:
            for i, (vid_id, metadata, distance) in enumerate(zip(
                results['ids'][0], results['metadatas'][0], results['distances'][0]
            )):
                similarity = 1 - distance
                with st.expander(f"Result {i+1}: {metadata.get('title', vid_id)} — Relevance: {similarity:.1%}", expanded=(i == 0)):
                    roi_key = metadata.get('intervention_key', '')
                    comparison_id = f"comparison_{roi_key}" if roi_key else vid_id
                    comparison_path = f"outputs/videos/{comparison_id}.mp4"
                    if os.path.exists(comparison_path):
                        _show_video(comparison_path)
                    else:
                        fallback_path = metadata.get('video_path', f"outputs/videos/{vid_id}.mp4")
                        if os.path.exists(fallback_path):
                            _show_video(fallback_path)
                        else:
                            st.info("No comparison video found for this result yet.")

                    c1, c2, c3 = st.columns(3)
                    if roi_key in roi_data_adjusted:
                        r = roi_data_adjusted[roi_key]
                        c1.metric('ROI', f"{r['roi']:.2f}x", f"+/-{r.get('u_precip', 0.5):.2f}")
                        c2.metric('Loss Avoided', _format_money_short(r['total_loss_avoided']))
                        c3.metric('Risk Reduction', f"{r['mean_risk_reduction']:.1%}")
                        c1.caption(
                            f"Range: {r.get('roi_lower', 0):.2f} - {r.get('roi_upper', 0):.2f}"
                        )

                        if metadata.get('has_tail_risk'):
                            st.warning(f"Tail-Risk Nodes Detected: {metadata.get('tail_risk_count', 'N/A')}")
if active_view == "Dashboard":
    # --- Metrics Dashboard ---
    section_header(
        "Compare",
        "Intervention comparison",
        "Read the payoff of each resilience strategy across ROI, avoided loss, and risk reduction before drilling into the training playback.",
    )

    roi_rows = []
    for key, data in roi_data_adjusted.items():
        roi_lower = float(data.get("roi_lower", 0.0))
        roi_upper = float(data.get("roi_upper", 0.0))
        roi_rows.append({
            "Intervention": data.get("name", key),
            "ROI (x)": float(data.get("roi", 0.0)),
            "ROI Range": f"{roi_lower:.2f} - {roi_upper:.2f}",
            "Loss Avoided ($M)": float(data.get("total_loss_avoided", 0.0)) / 1e6,
            "Mean Risk Reduction (%)": float(data.get("mean_risk_reduction", 0.0)) * 100.0,
            "Tail-Risk Nodes Neutralized": int(data.get("tail_risk_nodes_neutralized", 0)),
            "Eligible Footprint (%)": float(data.get("eligible_share", 0.0)) * 100.0,
        })

    roi_table = pd.DataFrame(roi_rows)
    if not roi_table.empty:
        roi_table = roi_table.sort_values("ROI (x)", ascending=False, ignore_index=True)

    view_cols = st.columns([1, 2])
    with view_cols[0]:
        compare_view = st.radio("View", ["Chart", "Table"], horizontal=True, label_visibility="visible")

    if compare_view == "Chart":
        if not roi_table.empty:
            show_top10 = st.checkbox("Top 10 only", value=True, help="Keep the chart short and video-ready.")
            filtered_table = roi_table.head(10) if show_top10 else roi_table
            chart_metric = st.selectbox(
                "Chart Metric",
                ["ROI (x)", "Loss Avoided ($M)", "Mean Risk Reduction (%)"],
                index=0,
            )
            chart_df = filtered_table.copy()
            chart_df["Intervention"] = pd.Categorical(
                chart_df["Intervention"],
                categories=chart_df.sort_values(chart_metric, ascending=True)["Intervention"],
                ordered=True,
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    y=alt.Y("Intervention:N", sort=None, title=""),
                    x=alt.X(f"{chart_metric}:Q", title=chart_metric),
                    color=alt.Color(f"{chart_metric}:Q", scale=alt.Scale(scheme="tealblues")),
                tooltip=["Intervention", chart_metric, "ROI Range"],
            )
                .properties(height=35 * len(chart_df))
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("ROI data not available yet. Run `python scripts/run_pipeline.py` first.")
    else:
        show_top10 = st.checkbox("Top 10 only", value=True, help="Keep the table compact and video-ready.")
        filtered_table = roi_table.head(10) if show_top10 else roi_table
        table_height = 42 * max(len(filtered_table), 1)
        st.dataframe(
            filtered_table,
            use_container_width=True,
            hide_index=True,
            height=min(table_height, 520),
            column_config={
                "ROI (x)": st.column_config.NumberColumn(format="%.2f"),
                "Loss Avoided ($M)": st.column_config.NumberColumn(format="%.1f"),
                "Mean Risk Reduction (%)": st.column_config.NumberColumn(format="%.1f"),
                "Eligible Footprint (%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

if active_view == "GNN Playback":
    training = load_training_history()

    # --- GNN Training Animation ---
    section_header(
        "Playback",
        "Interactive GNN training playback",
        "Scrub through epochs to see how node-level risk estimates stabilize and how the optimizer reshapes the graph-wide profile.",
    )

    if training is None:
        st.info("Training history not found yet. Re-run `python scripts/run_pipeline.py` to generate `outputs/roi/gnn_training_history.npz`.")
    else:
        total_epochs = int(len(training["epochs"]))
        if "training_epoch_idx" not in st.session_state:
            st.session_state.training_epoch_idx = total_epochs - 1
        if "training_playing" not in st.session_state:
            st.session_state.training_playing = False
        st.session_state.training_epoch_idx = min(max(int(st.session_state.training_epoch_idx), 0), total_epochs - 1)

        control_cols = st.columns([4, 1, 1, 1, 1.2])
        epoch_selected = control_cols[0].slider(
            "Epoch",
            min_value=1,
            max_value=total_epochs,
            value=st.session_state.training_epoch_idx + 1,
            disabled=st.session_state.training_playing,
            key="training_epoch_slider",
        )
        play_btn = control_cols[1].button(
            "Resume" if st.session_state.training_playing else "Play",
            use_container_width=True,
            key="training_play_btn",
        )
        pause_btn = control_cols[2].button("Pause", use_container_width=True, key="training_pause_btn")
        reset_btn = control_cols[3].button("Reset", use_container_width=True, key="training_reset_btn")
        playback_speed = control_cols[4].selectbox("Speed", ["Slow", "Medium", "Fast"], index=1, key="training_speed")
        highlight_targets = st.checkbox("Highlight hardest tail-risk targets", value=True)

        if play_btn:
            st.session_state.training_playing = True
        if pause_btn:
            st.session_state.training_playing = False
        if reset_btn:
            st.session_state.training_epoch_idx = 0
            st.session_state.training_playing = False
        elif not st.session_state.training_playing:
            st.session_state.training_epoch_idx = epoch_selected - 1

        speed_seconds = {"Slow": 0.18, "Medium": 0.07, "Fast": 0.02}[playback_speed]
        metrics_placeholder = st.empty()
        viz_placeholder = st.empty()

        render_training_frame(
            viz_placeholder,
            metrics_placeholder,
            training,
            st.session_state.training_epoch_idx,
            show_edges=True,
            highlight_targets=highlight_targets,
        )

        if st.session_state.training_playing:
            if st.session_state.training_epoch_idx >= total_epochs - 1:
                st.session_state.training_playing = False
            else:
                import time
                time.sleep(speed_seconds)
                st.session_state.training_epoch_idx += 1
                st.rerun()

if active_view == "Dashboard":
    ts = load_risk_timeseries()
    opportunity = load_opportunity_map()

    # --- Video Display ---
    section_header(
        "Watch",
        "Simulation videos",
        "Review the rendered outputs that feed the search index and communicate the intervention story visually.",
    )

    video_dir = Path("outputs/videos")
    if video_dir.exists():
        videos = sorted(video_dir.glob("*.mp4"), key=lambda p: p.name.lower())
        if videos:
            comparison_videos = [p for p in videos if p.name.startswith("comparison_")]
            grid_videos = [
                p
                for p in videos
                if p.stem.startswith("interventions_")
                or p.stem.startswith("megavideo_")
                or "grid" in p.stem
                or "megavideo" in p.stem
            ]
            core_videos = [
                p
                for p in videos
                if p.stem in {"baseline_risk", "tail_risk_escalation", "climate_classification_shift"}
            ]

            video_type = st.selectbox(
                "Video type",
                ["Comparison", "Core Maps", "Grid"],
                index=0,
                help="Use this dropdown instead of tabs to avoid horizontal scrolling when many videos exist.",
            )

            if video_type == "Comparison":
                if not comparison_videos:
                    st.info("No comparison videos found yet. Render them via `bash scripts/submit_render_comparisons.sh`.")
                else:
                    def _cmp_label(p: Path) -> str:
                        key = p.stem[len("comparison_") :] if p.stem.startswith("comparison_") else p.stem
                        name = INTERVENTIONS.get(key, {}).get("name") or key.replace("_", " ").title()
                        return f"{name}"

                    options = { _cmp_label(p): p for p in sorted(comparison_videos, key=_cmp_label) }
                    label = st.selectbox("Select intervention", list(options.keys()), index=0)
                    st.caption(str(options[label]))
                    _show_video(str(options[label]))

            elif video_type == "Core Maps":
                if not core_videos:
                    st.info("No core map videos found yet. Run `python scripts/run_pipeline.py`.")
                else:
                    friendly = {
                        "baseline_risk": "Baseline Risk",
                        "tail_risk_escalation": "Tail-Risk Escalation",
                        "climate_classification_shift": "Koppen-Geiger Shift",
                    }
                    options = {}
                    for p in core_videos:
                        label = friendly.get(p.stem, p.stem.replace("_", " ").title())
                        if label in options:
                            label = f"{label} ({p.name})"
                        options[label] = p
                    label = st.selectbox("Select map", list(options.keys()), index=0)
                    st.caption(str(options[label]))
                    _show_video(str(options[label]))

            else:  # Grid
                if not grid_videos:
                    st.info(
                        "No grid/mega videos found yet.\n\n"
                        "Render one with:\n"
                        "`python scripts/render_megavideo_from_npz.py --mode grid --out outputs/videos/interventions_grid.mp4 --ncols 6`"
                    )
                else:
                    options = {p.stem.replace("_", " ").title(): p for p in grid_videos}
                    label = st.selectbox("Select grid video", list(options.keys()), index=0)
                    st.caption(str(options[label]))
                    _show_video(str(options[label]))
        else:
            st.info("No videos generated yet. Run `python scripts/run_pipeline.py`.")
    else:
        st.info('Output directory not found. Run the pipeline first.')

    # --- Quantitative Risk Chart ---
    section_header(
        "Monitor",
        "Risk over time",
        "Track how baseline and intervention pathways diverge across the full simulation horizon.",
    )
    if ts is None:
        st.info("Risk time series not found yet. Re-run the pipeline to generate `outputs/roi/risk_timeseries.json`.")
    else:
        intervention_keys = [k for k in ts.keys() if k not in {"years", "baseline"}]
        category_map = {"All": []}
        for key in intervention_keys:
            category = INTERVENTIONS.get(key, {}).get("category", "other").replace("_", " ").title()
            category_map.setdefault(category, []).append(key)

        category_map["All"] = sorted(intervention_keys)
        category_options = ["All"] + sorted([c for c in category_map.keys() if c != "All"])
        category_choice = st.selectbox("Category", category_options, index=0)

        scoped_keys = category_map.get(category_choice, [])
        scoped_labels = ["Baseline"]
        scoped_lookup = {"Baseline": None}
        for key in scoped_keys:
            label = INTERVENTIONS.get(key, {}).get("name", key.replace("_", " ").title())
            scoped_labels.append(label)
            scoped_lookup[label] = key

        choice = st.selectbox("Intervention", scoped_labels, index=0)
        chosen_key = scoped_lookup.get(choice)

        fig = build_risk_timeseries_figure(ts, "p95", chosen_key, choice if chosen_key else None)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --- Tail Risk Map ---
    section_header(
        "Locate",
        "Tail-risk escalation map",
        "Scan the geography of exposure and opportunity. Red overlays indicate nodes exceeding the modelâ€™s extreme-regime threshold.",
    )
    if opportunity is not None:
        with st.expander("Interactive map (optional)", expanded=False):
            st.caption("Optional 2D/3D basemap view. The default below is the static ROI PNG.")
            map_mode = st.selectbox(
                "Interactive map mode",
                [
                    "2D ROI map (basemap)",
                    "3D ROI extrusion (slower)",
                ],
                index=0,
                key="sentree_map_mode",
            )
            st.markdown("Hover any cell to see a real-world label (nearest city + distance).")
            try:
                import pydeck as pdk

                df_pts, (vmin, vmax) = _opportunity_points(opportunity)

                with st.expander("Map debug (lat/lon ranges)"):
                    lats_dbg = np.asarray(opportunity["lats"], dtype=np.float64)
                    lons_raw_dbg = np.asarray(opportunity["lons"], dtype=np.float64)
                    lons_dbg = _wrap_lon_180(lons_raw_dbg)
                    st.write(
                        {
                            "lat_min": float(np.nanmin(lats_dbg)),
                            "lat_max": float(np.nanmax(lats_dbg)),
                            "lon_min_raw": float(np.nanmin(lons_raw_dbg)),
                            "lon_max_raw": float(np.nanmax(lons_raw_dbg)),
                            "lon_min_wrapped": float(np.nanmin(lons_dbg)),
                            "lon_max_wrapped": float(np.nanmax(lons_dbg)),
                            "n_points": int(len(df_pts)),
                        }
                    )

                max_points = 50_000
                if len(df_pts) > max_points:
                    df_pts = df_pts.sample(max_points, random_state=0)

                lat_span = float(df_pts["lat"].max() - df_pts["lat"].min())
                lon_span = float(df_pts["lon"].max() - df_pts["lon"].min())
                zoom = _suggest_pydeck_zoom(lat_span, lon_span)
                pitch = 0 if map_mode == "2D ROI map (basemap)" else (0 if zoom <= 2.2 else 50)

                view = pdk.ViewState(
                    latitude=float(df_pts["lat"].median()),
                    longitude=float(df_pts["lon"].median()),
                    zoom=zoom,
                    pitch=pitch,
                )

                cell_size_m = _approx_cell_size_m(opportunity["lats"], opportunity["lons"])
                extruded = map_mode == "3D ROI extrusion (slower)"
                elevation_scale = 4000.0 if extruded else 1.0
                layer_cells = pdk.Layer(
                    "GridCellLayer",
                    data=df_pts,
                    get_position=["lon", "lat"],
                    get_elevation="value",
                    elevation_scale=elevation_scale,
                    cell_size=cell_size_m,
                    extruded=extruded,
                    pickable=True,
                    get_fill_color="color",
                    opacity=0.68 if not extruded else 0.85,
                )

                tooltip = {
                    "text": (
                        "Avoided damage potential: {value}\n"
                        "Nearest: {nearest_city} (~{nearest_km} km)"
                    )
                }

                deck = pdk.Deck(
                    layers=[layer_cells],
                    initial_view_state=view,
                    tooltip=tooltip,
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                )
                st.pydeck_chart(deck, width='stretch')
                st.caption(f"Color scale uses min/max of the opportunity grid: vmin={vmin:.4f}, vmax={vmax:.4f}.")
            except Exception as e:
                st.info(f"Interactive map unavailable ({e}).")

    tail_risk_img = 'outputs/tail_risk_map.png'
    if os.path.exists(tail_risk_img):
        st.image(tail_risk_img, width='stretch')
    else:
        st.info('Tail-risk map not generated yet.')

if active_view == "Math":
    render_math_view()

# --- Footer ---
st.divider()
st.caption('SenTree - Resilience ROI Dashboard | ML@Purdue Catapult Hackathon')

