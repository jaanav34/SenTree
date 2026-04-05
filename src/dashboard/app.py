"""SenTree Dashboard — Streamlit UI."""
import os
import sys
import json
import time
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from sentree_venv import ensure_venv

ensure_venv()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection

st.set_page_config(page_title='SenTree — Resilience ROI Dashboard', layout='wide')

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

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 30%),
            radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 28%),
            linear-gradient(180deg, #f1eee2 0%, #e4efe8 54%, #f6f3ea 100%);
        color: var(--sentree-ink);
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

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sentree-kicker">Mission Control</div>', unsafe_allow_html=True)
    st.markdown("## Scenario Controls")
    st.markdown("Tune the climate lens, intervention package, and search posture before diving into simulations.")
    scenario = st.selectbox('Climate Scenario', ['SSP3-7.0 (High Emissions)', 'SSP1-2.6 (Low Emissions)'])
    intervention = st.selectbox('Intervention', ['Coastal Mangrove Restoration', 'Regenerative Agriculture', 'Both'])
    st.divider()
    surface_card("Deployment Region", "SE Asia coastal network with tail-risk emphasis on dense coastal and agricultural nodes.")
    surface_card("Forecast Horizon", "2015-2100 scenario window with intervention comparisons and searchable video outputs.")

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

    # 1° latitude ~= 111 km (good enough for UI sizing)
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

    # WebMercator (deck.gl basemaps) only supports latitudes up to about ±85.051°.
    # Global ISIMIP grids often include points up to ~±89°, which will render "off-map"
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
    ax_risk.legend(frameon=False, loc="upper left")

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


def build_risk_timeseries_figure(ts, metric):
    years = np.array(ts["years"])
    fig, ax = plt.subplots(figsize=(12.8, 4.6))

    series = [("Baseline", ts["baseline"][metric], "#17342f")]
    if "mangrove_restoration" in ts:
        series.append(("Mangrove Restoration", ts["mangrove_restoration"][metric], "#0f766e"))
    if "regenerative_agriculture" in ts:
        series.append(("Regenerative Agriculture", ts["regenerative_agriculture"][metric], "#b45309"))

    for name, values, color in series:
        values_arr = np.array(values)
        ax.plot(years, values_arr, linewidth=2.4, color=color, label=name)
        ax.scatter([years[-1]], [values_arr[-1]], color=color, s=42, zorder=3)

    ax.set_title("Systemic Risk Trajectory", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric.upper())
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, loc="upper left", ncols=len(series))
    ax.set_facecolor("#fffdf7")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.patch.set_facecolor("#fffaf2")
    return fig


top_intervention = max(roi_data.values(), key=lambda item: item.get("roi", 0.0))
training_history_path = "outputs/roi/gnn_training_history.npz"
training_status = "Training snapshots ready" if os.path.exists(training_history_path) else "Run pipeline to generate playback"
video_count = len([f for f in os.listdir('outputs/videos') if f.endswith('.mp4')]) if os.path.exists('outputs/videos') else 0

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
            <div class="sentree-badge">{training_status}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_cols = st.columns(4)
with hero_cols[0]:
    kpi_card("Best ROI", f"{top_intervention.get('roi', 0):.2f}x", top_intervention["name"])
with hero_cols[1]:
    kpi_card("Loss Avoided", f"${top_intervention.get('total_loss_avoided', 0)/1e9:.1f}B", "Top intervention impact")
with hero_cols[2]:
    kpi_card("Playback", "Ready" if os.path.exists(training_history_path) else "Missing", "Training view availability")
with hero_cols[3]:
    kpi_card("Rendered Videos", str(video_count), "Simulation clips available")

active_view = st.radio(
    "View",
    ["Dashboard", "GNN Playback"],
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

if active_view == "Dashboard":
    # --- Search Results ---
    if search_btn and query:
        surface_card("Search Results", "Ranked video matches using vector search and intervention metadata.")

        try:
            from src.embedding.vectordb import VideoSearchDB
            db = VideoSearchDB()

            if db.count() > 0:
                results = db.query(query, n_results=3)

                for i, (vid_id, metadata, distance) in enumerate(zip(
                    results['ids'][0], results['metadatas'][0], results['distances'][0]
                )):
                    similarity = 1 - distance
                    with st.expander(f'Result {i+1}: {metadata.get("title", vid_id)} — Relevance: {similarity:.1%}', expanded=(i == 0)):
                        video_path = metadata.get('video_path', f'outputs/videos/{vid_id}.mp4')
                        if os.path.exists(video_path):
                            _show_video(video_path)

                        c1, c2, c3 = st.columns(3)
                        roi_key = metadata.get('intervention_key', '')
                        if roi_key in roi_data:
                            r = roi_data[roi_key]
                            c1.metric('ROI', f"{r['roi']:.2f}x", f"+/-{r.get('u_precip', 0.5):.2f}")
                            c2.metric('Loss Avoided', f"${r['total_loss_avoided']/1e9:.1f}B")
                            c3.metric('Risk Reduction', f"{r['mean_risk_reduction']:.1%}")

                        if metadata.get('has_tail_risk'):
                            st.warning(f"Tail-Risk Nodes Detected: {metadata.get('tail_risk_count', 'N/A')}")
            else:
                st.info('No videos indexed yet. Run `python scripts/index_videos.py` first.')

        except Exception as e:
            st.error(f'Search error: {e}')
            st.info('Showing demo results instead.')

    # --- Metrics Dashboard ---
    section_header(
        "Compare",
        "Intervention comparison",
        "Read the payoff of each resilience strategy across ROI, avoided loss, and risk reduction before drilling into the training playback.",
    )

    cols = st.columns(len(roi_data))
    for i, (key, data) in enumerate(roi_data.items()):
        with cols[i]:
            surface_card(
                data['name'],
                "Financial and systemic impact snapshot for the currently indexed intervention pathway.",
            )
            st.metric('Resilience ROI', f"{data['roi']:.2f}x",
                       help=f"Range: {data.get('roi_lower', 0):.2f} - {data.get('roi_upper', 0):.2f}")
            st.metric('Total Loss Avoided', f"${data.get('total_loss_avoided', 0)/1e9:.1f}B")
            st.metric('Mean Risk Reduction', f"{data.get('mean_risk_reduction', 0):.1%}")

            tail_count = data.get('tail_risk_nodes_neutralized', 0)
            if tail_count > 0:
                st.error(f'Tail-Risk Nodes Neutralized: {tail_count}')

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

        control_cols = st.columns([4, 1, 1, 1, 1.2, 1.2])
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
        show_edges = control_cols[5].checkbox("Show graph", value=True, key="training_show_graph")
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
            show_edges=show_edges,
            highlight_targets=highlight_targets,
        )

        if st.session_state.training_playing:
            if st.session_state.training_epoch_idx >= total_epochs - 1:
                st.session_state.training_playing = False
            else:
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

    video_dir = 'outputs/videos'
    if os.path.exists(video_dir):
        videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if videos:
            tabs = st.tabs([v.replace('.mp4', '').replace('_', ' ').title() for v in videos])
            for tab, vid in zip(tabs, videos):
                with tab:
                    _show_video(os.path.join(video_dir, vid))
        else:
            st.info('No videos generated yet. Run `python scripts/run_pipeline.py`.')
    else:
        st.info('Output directory not found. Run the pipeline first.')

    # --- Math & Methodology Tab ---
    st.divider()
    section_header(
        "Explain",
        "Technical deep-dive",
        "Inspect the mathematical assumptions behind tail-risk escalation, ROI estimation, and the graph model itself.",
    )

    math_tab, playground_tab = st.tabs(['📐 Mathematical Foundations', '🎮 Interactive Playground'])

    with math_tab:
        st.subheader('1. Tail-Risk Escalation (Gurjar & Camp 2026)')
        st.markdown(r"""
    The tail-risk engine identifies nodes where climate volatility and momentum intersect to create "regime shifts."
    
    **A. EWMA Smoothing (Intensity):**
    First, raw signals $X(t)$ (temperature/precipitation) are smoothed to suppress high-frequency noise:
    $$\lambda(t) = \\alpha X(t) + (1 - \\alpha)\lambda(t-1)$$
    where $\\alpha = 0.3$ is the decay factor.
    
    **B. Standardized Momentum:**
    Momentum captures the acceleration of the climate signal, standardized by local rolling volatility:
    $$m(t) = \\frac{\lambda(t) - \lambda(t-1)}{\sigma_w(t) + \\epsilon}$$
    where $\sigma_w(t)$ is the rolling standard deviation over window $w$.
    
    **C. Rolling Volatility:**
    Volatility measures the stability of the signal:
    $$v(t) = \\sqrt{\\frac{1}{w} \sum_{i=t-w+1}^{t} [m(i) - \\bar{m}_w]^2}$$
    
    D. Hawkes Self-Excitation:
    To capture "clusters" of extreme events, we add a Hawkes process intensity:
    $$\lambda^*(t) = \mu + \sum_{t_i < t} \\beta e^{-\gamma(t - t_i)}$$
    Nodes exceeding the 95th percentile of the composite score are flagged as **Tail-Risk Escalation** zones.

    **E. Köppen-Geiger Climate Classification:**
    We model climate shifts by classifying each node annually based on monthly temperature and precipitation thresholds. 
    Groups include:
    *   **Group A (Tropical):** $T_{min} \ge 18^\circ C$
    *   **Group B (Dry):** $P_{ann} < 10 \times P_{thresh}$
    *   **Group C (Temperate):** $0^\circ C < T_{min} < 18^\circ C$
    *   **Group D (Continental):** $T_{min} \le 0^\circ C$
    *   **Group E (Polar):** $T_{max} < 10^\circ C$
    """)

    
        st.subheader('2. Resilience ROI & Economic Exposure (Ito 2020)')
        st.markdown(r"""
    **Avoided Damage Potential (The "Green Shades"):**
    The green shades on our map represent the **Resilience Opportunity Index (ROI)**, which is the potential damage avoided by an intervention.
    
    **A. Loss Proxy ($L$):**
    Loss is modeled as the intersection of climate risk ($R$), GDP ($G$), and Population ($P$):
    $$L_{node} = R_{score} \\times (G_{norm} \\times P_{norm}) \\times S$$
    where $S$ is a regional scaling factor.
    
    **B. Resilience ROI:**
    The return is the sum of discounted avoided losses over a 10-year horizon:
    $$ROI_{resilience} = \\frac{\sum_{t=1}^{10} (L_{baseline} - L_{intervention}) \\times (1+r)^{-t}}{Cost}$$
    where $r$ is the discount rate (default 5%).
    
    **C. Multi-Source Uncertainty:**
    Following **Ito et al. (2020)**, we compute uncertainty via quadrature:
    $$U_{total} = \\sqrt{U_{precip}^2 + U_{model}^2 + U_{scenario}^2}$$
    """)

        st.subheader('3. GNN Risk Architecture')
        st.markdown(r"""
    Our GNN uses a **Graph Attention Network (GAT)** to propagate risk through geographic and economic links:
    $$h_i^{(l+1)} = \\sigma\left( \sum_{j \in \\mathcal{N}(i)} \\alpha_{ij} \mathbf{W} h_j^{(l)} \\right)$$
    where attention $\\alpha_{ij}$ is computed based on distance and feature similarity.
    """)

    with playground_tab:
        st.subheader('Risk Simulation Playground')
        st.markdown('Adjust parameters to see how they impact the ROI calculation logic.')
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_alpha = st.slider('EWMA Decay ($\\alpha$)', 0.05, 0.95, 0.30)
            p_discount = st.slider('Discount Rate ($r$)', 0.01, 0.15, 0.05)
        with col_p2:
            p_exposure = st.slider('Economic Exposure Scaling', 0.5, 5.0, 1.0)
            p_threshold = st.slider('Tail-Risk Percentile', 80, 99, 95)
        
        # Mock calculation for playground
        base_val = 100.0
        reduction = 15.0 * p_exposure
        discounted_val = sum([reduction / (1 + p_discount)**t for t in range(10)])
        
        st.info(f"**Theoretical Outcome:** An intervention reducing risk by {reduction:.1f}% would yield a total discounted loss avoidance of **${discounted_val:.2f}B** over 10 years.")
        
        st.markdown("""
    **Statistical Implications:**
    *   **Higher $\\alpha$:** Makes the system more sensitive to recent shocks (higher volatility).
    *   **Lower $r$:** Increases the present value of future resilience (favoring long-term projects like Mangroves).
    *   **Higher Threshold:** Focuses only on the most extreme "black swan" events.
    """)

    # --- Quantitative Risk Chart ---
    section_header(
        "Monitor",
        "Risk over time",
        "Track how baseline and intervention pathways diverge across the full simulation horizon.",
    )
    if ts is None:
        st.info("Risk time series not found yet. Re-run the pipeline to generate `outputs/roi/risk_timeseries.json`.")
    else:
        metric = st.selectbox("Metric", ["mean", "p95", "max"], index=0)
        fig = build_risk_timeseries_figure(ts, metric)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --- Tail Risk Map ---
    section_header(
        "Locate",
        "Tail-risk escalation map",
        "Scan the geography of exposure and opportunity. Red overlays indicate nodes exceeding the model’s extreme-regime threshold.",
    )
    if opportunity is not None:
        st.markdown("**Interactive 3D Opportunity Overlay** (hover any cell to see a real-world label).")
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

            # Optional downsample for performance if needed.
            # 16k points (global coarsen=4) is fine; only downsample when extremely large.
            max_points = 50_000
            if len(df_pts) > max_points:
                df_pts = df_pts.sample(max_points, random_state=0)

            lat_span = float(df_pts["lat"].max() - df_pts["lat"].min())
            lon_span = float(df_pts["lon"].max() - df_pts["lon"].min())
            zoom = _suggest_pydeck_zoom(lat_span, lon_span)
            pitch = 0 if zoom <= 2.2 else 50

            view = pdk.ViewState(
                latitude=float(df_pts["lat"].median()),
                longitude=float(df_pts["lon"].median()),
                zoom=zoom,
                pitch=pitch,
            )

            # Grid cells (3D extruded)
            cell_size_m = _approx_cell_size_m(opportunity["lats"], opportunity["lons"])
            elevation_scale = 4000.0  # tune for visibility
            layer_cells = pdk.Layer(
                "GridCellLayer",
                data=df_pts,
                get_position=["lon", "lat"],
                get_elevation="value",
                elevation_scale=elevation_scale,
                cell_size=cell_size_m,
                extruded=True,
                pickable=True,
                get_fill_color="color",
            )

            # Tail-risk flags (red dots overlaid)
            flagged = df_pts[df_pts["tail_flag"]].copy()
            layer_flags = pdk.Layer(
                "ScatterplotLayer",
                data=flagged,
                get_position=["lon", "lat"],
                get_radius=max(10_000, int(round(0.45 * cell_size_m))),
                get_fill_color=[230, 30, 30, 190],
                pickable=True,
            )

            tooltip = {
                "text": (
                    "Lon: {lon}\nLat: {lat}\n"
                    "Avoided damage potential: {value}\n"
                    "Tail-risk flagged: {tail_flag}\n"
                    "Nearest: {nearest_city} (~{nearest_km} km)"
                )
            }

            deck = pdk.Deck(
                layers=[layer_cells, layer_flags],
                initial_view_state=view,
                tooltip=tooltip,
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            )
            st.pydeck_chart(deck, width='stretch')
            st.caption(f"Color scale uses min/max of the opportunity grid: vmin={vmin:.4f}, vmax={vmax:.4f}.")
        except Exception as e:
            st.info(f"Interactive map unavailable ({e}). Showing static map below.")

    tail_risk_img = 'outputs/tail_risk_map.png'
    if os.path.exists(tail_risk_img):
        st.image(tail_risk_img, width='stretch')
    else:
        st.info('Tail-risk map not generated yet.')

# --- Footer ---
st.divider()
st.caption('SenTree — Resilience ROI Dashboard | ML@Purdue Catapult Hackathon')
