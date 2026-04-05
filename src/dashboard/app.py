"""SenTree Dashboard â€” Streamlit UI."""
import os
import sys
import json
import base64
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

from src.simulation.interventions import INTERVENTIONS, climate_fit_summary

st.set_page_config(page_title='SenTree - Resilience ROI Dashboard', layout='wide', initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;700;800;900&family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,300;9..144,700;9..144,900&display=swap');

    :root {
        --sentree-ink: #0f1f1c;
        --sentree-ink-soft: #243b35;
        --sentree-ink-muted: #4a6159;
        --sentree-accent: #0d9488;
        --sentree-accent-dark: #0f766e;
        --sentree-accent-warm: #c2690a;
        --sentree-accent-warm-soft: rgba(194, 105, 10, 0.12);
        --sentree-glow: rgba(13, 148, 136, 0.18);
        --sentree-glow-warm: rgba(194, 105, 10, 0.14);
        --sentree-card-top: rgba(252, 249, 242, 0.97);
        --sentree-card-bottom: rgba(240, 248, 244, 0.95);
        --sentree-card-border: rgba(15, 50, 42, 0.09);
        --sentree-sidebar-bg: linear-gradient(160deg, #0c1e1a 0%, #122920 50%, #0e2235 100%);
        --sentree-sidebar-ink: #ede8d8;
        --sentree-sidebar-muted: #c8c0a8;
        --sentree-sidebar-field: rgba(248, 244, 234, 0.94);
        --sentree-sidebar-border: rgba(237, 232, 216, 0.14);
        --radius-card: 24px;
        --radius-pill: 999px;
        --shadow-card: 0 2px 4px rgba(10,30,24,0.04), 0 8px 24px rgba(10,30,24,0.07), 0 24px 48px rgba(10,30,24,0.05);
        --shadow-card-hover: 0 4px 8px rgba(10,30,24,0.06), 0 16px 40px rgba(10,30,24,0.12), 0 40px 72px rgba(10,30,24,0.07);
        --shadow-hero: 0 1px 2px rgba(10,30,24,0.04), 0 8px 32px rgba(10,30,24,0.08), 0 32px 80px rgba(10,30,24,0.08);
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(13,148,136,0); }
        50%       { box-shadow: 0 0 0 6px rgba(13,148,136,0.10); }
    }
    @keyframes spinSlow {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }

    header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    div[data-testid="stToolbar"]   { visibility: hidden; height: 0; }

    /* ── Page background ─────────────────────────────────────────── */
    .stApp {
        background:
            radial-gradient(ellipse 60% 45% at 8% 0%,   rgba(13,148,136,0.13) 0%, transparent 100%),
            radial-gradient(ellipse 50% 40% at 95% 5%,  rgba(194,105,10,0.10) 0%, transparent 100%),
            radial-gradient(ellipse 80% 60% at 50% 100%,rgba(13,148,136,0.06) 0%, transparent 100%),
            linear-gradient(170deg, #eeeade 0%, #e2ede7 38%, #eceae0 70%, #f0ede3 100%);
        color: var(--sentree-ink);
        overflow-x: hidden;
        font-family: "Cabinet Grotesk", "Manrope", system-ui, sans-serif;
        font-size: 15px;
        letter-spacing: -0.01em;
    }

    /* Subtle noise texture overlay */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
        background-size: 180px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.6;
    }

    html, body { overflow-x: hidden; }

    .block-container {
        max-width: 1340px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        animation: fadeSlideUp 0.5s ease both;
    }

    /* ── Hero ────────────────────────────────────────────────────── */
    .sentree-hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem 1.8rem;
        border-radius: 32px;
        background:
            radial-gradient(circle at 88% 15%, rgba(255,255,255,0.42) 0%, transparent 22%),
            radial-gradient(circle at 6% 80%,  rgba(13,148,136,0.10) 0%, transparent 26%),
            linear-gradient(138deg, rgba(255,253,247,0.97) 0%, rgba(230,243,238,0.94) 100%);
        border: 1px solid rgba(15,118,110,0.14);
        box-shadow: var(--shadow-hero);
        margin-bottom: 1.6rem;
        animation: fadeSlideUp 0.5s 0.05s ease both;
    }

    /* Decorative arc in top-right */
    .sentree-hero::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        border-radius: 50%;
        border: 1.5px solid rgba(13,148,136,0.12);
        pointer-events: none;
    }

    /* Warm orb bottom-right */
    .sentree-hero::after {
        content: "";
        position: absolute;
        bottom: -80px; right: 8%;
        width: 320px; height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(194,105,10,0.11), transparent 65%);
        pointer-events: none;
    }

    .sentree-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.78rem 0.32rem 0.62rem;
        border-radius: var(--radius-pill);
        background: rgba(13,148,136,0.10);
        border: 1px solid rgba(13,148,136,0.18);
        color: var(--sentree-accent-dark);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }
    .sentree-kicker::before {
        content: "";
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--sentree-accent);
        display: inline-block;
        animation: pulseGlow 2.4s ease-in-out infinite;
    }

    .sentree-hero h1 {
        margin: 0.9rem 0 0.5rem;
        font-size: clamp(2.4rem, 4vw, 3.4rem);
        line-height: 0.93;
        font-family: "Fraunces", Georgia, serif;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #0d2a24 0%, #0f766e 55%, #0d2a24 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 6s linear infinite;
    }

    .sentree-hero p {
        margin: 0;
        max-width: 52rem;
        color: var(--sentree-ink-soft);
        font-size: 1.02rem;
        line-height: 1.6;
        font-weight: 400;
    }

    .sentree-hero-grid {
        display: flex;
        gap: 1.4rem;
        align-items: flex-start;
        justify-content: space-between;
    }

    .sentree-hero-copy { flex: 1 1 auto; min-width: 0; }

    .sentree-hero-logo img {
        width: 116px;
        height: auto;
        border-radius: 18px;
        border: 1px solid rgba(15,118,110,0.16);
        box-shadow: 0 8px 28px rgba(10,30,24,0.14);
        background: rgba(255,255,255,0.85);
        padding: 0.3rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .sentree-hero-logo img:hover {
        transform: scale(1.04) rotate(-1deg);
        box-shadow: 0 12px 36px rgba(10,30,24,0.20);
    }

    .sentree-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.2rem;
    }

    .sentree-badge {
        padding: 0.42rem 0.78rem;
        border-radius: var(--radius-pill);
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(15,50,42,0.10);
        color: #173c34;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        backdrop-filter: blur(6px);
        transition: background 0.2s ease, transform 0.2s ease;
    }
    .sentree-badge:hover {
        background: rgba(255,255,255,0.90);
        transform: translateY(-1px);
    }

    /* ── Section header ──────────────────────────────────────────── */
    .sentree-section { margin: 1.5rem 0 0.5rem; }

    .sentree-section-label {
        color: var(--sentree-accent);
        font-size: 0.70rem;
        font-weight: 900;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.28rem;
        font-family: "DM Mono", monospace;
    }

    .sentree-section h2 {
        margin: 0;
        font-size: clamp(1.6rem, 2.5vw, 2.1rem);
        font-family: "Fraunces", Georgia, serif;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.05;
    }

    .sentree-section p {
        margin: 0.32rem 0 0;
        color: var(--sentree-ink-muted);
        max-width: 54rem;
        font-size: 0.96rem;
        line-height: 1.55;
    }

    /* ── Cards ───────────────────────────────────────────────────── */
    .sentree-card {
        padding: 1.25rem 1.35rem;
        border-radius: var(--radius-card);
        background: linear-gradient(160deg, rgba(254,251,244,0.97) 0%, rgba(239,248,244,0.95) 100%);
        border: 1px solid var(--sentree-card-border);
        box-shadow: var(--shadow-card);
        margin-bottom: 1rem;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .sentree-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--sentree-accent), var(--sentree-accent-warm), var(--sentree-accent));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .sentree-card:hover { box-shadow: var(--shadow-card-hover); transform: translateY(-1px); }
    .sentree-card:hover::before { opacity: 1; }

    .sentree-card h3 {
        margin: 0;
        font-size: 1.04rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .sentree-card p {
        margin: 0.4rem 0 0;
        color: var(--sentree-ink-soft);
        font-size: 0.94rem;
        line-height: 1.55;
    }

    /* ── KPI cards ───────────────────────────────────────────────── */
    .sentree-kpi {
        padding: 1.2rem 1.3rem;
        border-radius: var(--radius-card);
        background: linear-gradient(150deg, rgba(255,250,241,0.98) 0%, rgba(233,246,240,0.95) 100%);
        border: 1px solid var(--sentree-card-border);
        box-shadow: var(--shadow-card);
        min-height: 122px;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .sentree-kpi::after {
        content: "";
        position: absolute;
        bottom: -30px; right: -30px;
        width: 90px; height: 90px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(13,148,136,0.10), transparent 70%);
        pointer-events: none;
    }
    .sentree-kpi:hover { box-shadow: var(--shadow-card-hover); transform: translateY(-2px); }

    .sentree-kpi-label {
        color: var(--sentree-ink-muted);
        font-size: 0.71rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        font-family: "DM Mono", monospace;
    }

    .sentree-kpi-value {
        margin-top: 0.5rem;
        color: #0f2a24;
        font-size: 2.1rem;
        font-weight: 900;
        line-height: 1;
        font-family: "Fraunces", Georgia, serif;
        letter-spacing: -0.03em;
    }

    .sentree-kpi-sub {
        margin-top: 0.45rem;
        color: var(--sentree-ink-muted);
        font-size: 0.87rem;
    }

    /* ── Global type ─────────────────────────────────────────────── */
    h1, h2, h3 {
        color: #0f2a24;
        letter-spacing: -0.025em;
        font-family: "Fraunces", Georgia, serif;
    }
    h3 { font-family: "Cabinet Grotesk", system-ui, sans-serif; }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] > div {
        background: var(--sentree-sidebar-bg);
        border-right: 1px solid rgba(237,232,216,0.10);
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

    /* ── Metric widgets ──────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(150deg, var(--sentree-card-top), var(--sentree-card-bottom));
        border: 1px solid rgba(15,50,42,0.12);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow-card);
        transition: box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { box-shadow: var(--shadow-card-hover); }

    div[data-testid="stMetricLabel"] p {
        color: var(--sentree-ink-muted) !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        font-family: "DM Mono", monospace;
    }

    div[data-testid="stMetricValue"] {
        color: #0f2a24 !important;
        font-weight: 900;
        font-family: "Fraunces", Georgia, serif;
        letter-spacing: -0.02em;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--sentree-accent) !important;
        font-weight: 700;
    }

    /* ── Buttons ─────────────────────────────────────────────────── */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: var(--radius-pill);
        border: 1px solid rgba(15,50,42,0.20);
        background: linear-gradient(135deg, #122920, #0f766e);
        color: #f4eedc;
        font-weight: 800;
        font-family: "Cabinet Grotesk", system-ui, sans-serif;
        letter-spacing: -0.01em;
        box-shadow: 0 4px 12px rgba(13,118,110,0.22), 0 1px 3px rgba(0,0,0,0.10);
        transition: all 0.22s ease;
    }
    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #0f766e, #0d9488);
        border-color: rgba(13,148,136,0.35);
        box-shadow: 0 6px 20px rgba(13,148,136,0.30), 0 2px 6px rgba(0,0,0,0.10);
        transform: translateY(-1px);
    }
    div.stButton > button p,
    div[data-testid="stFormSubmitButton"] button p {
        color: inherit !important;
        font-weight: 800;
    }

    /* ── Inputs & selects ────────────────────────────────────────── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background: rgba(252,249,242,0.94);
        border-color: rgba(15,50,42,0.16) !important;
        color: #0f2a24;
        border-radius: 12px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="base-input"]:focus-within {
        border-color: rgba(13,148,136,0.45) !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.10);
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="base-input"] input {
        color: #0f2a24 !important;
        font-family: "Cabinet Grotesk", system-ui, sans-serif;
    }

    div[data-testid="stWidgetLabel"] *,
    label[data-baseweb="checkbox"] span,
    .stSelectbox label,
    .stSlider label,
    .stTextInput label {
        color: #1a3d35 !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
        background: var(--sentree-sidebar-field);
        border-color: var(--sentree-sidebar-border);
        color: #0f2a24;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="base-input"] input {
        color: #0f2a24 !important;
    }

    section[data-testid="stSidebar"] .sentree-card {
        background: linear-gradient(160deg, rgba(255,251,245,0.99), rgba(238,247,244,0.97));
        border-color: rgba(15,50,42,0.12);
        box-shadow: 0 8px 24px rgba(8,18,22,0.16);
    }

    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] .sentree-card h3 { color: #0f2a24 !important; }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] .sentree-card p  { color: #3d5850 !important; }

    section[data-testid="stSidebar"] div.stButton > button {
        background: rgba(237,232,216,0.10);
        border-color: rgba(237,232,216,0.20);
        color: var(--sentree-sidebar-ink);
        box-shadow: none;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(13,148,136,0.28);
        border-color: rgba(13,148,136,0.40);
    }

    section[data-testid="stSidebar"] [data-baseweb="checkbox"] > div {
        color: var(--sentree-sidebar-ink);
    }

    /* ── Slider ──────────────────────────────────────────────────── */
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"] { color: #4a6159; }

    div[data-baseweb="slider"] [role="slider"] {
        background: var(--sentree-accent);
        border-color: var(--sentree-accent);
        box-shadow: 0 0 0 4px rgba(13,148,136,0.15);
    }
    div[data-baseweb="slider"] > div > div { background: rgba(13,148,136,0.20); }

    /* ── Misc Streamlit elements ─────────────────────────────────── */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCaptionContainer"] { color: var(--sentree-ink-soft); }

    div[data-testid="stAlert"] { border-radius: 16px; }

    div[data-testid="stExpander"] {
        border-radius: 20px;
        border: 1px solid var(--sentree-card-border);
        background: rgba(252,249,242,0.82);
        overflow: hidden;
        transition: box-shadow 0.2s ease;
    }
    div[data-testid="stExpander"]:hover { box-shadow: 0 6px 20px rgba(10,30,24,0.08); }
    div[data-testid="stExpander"] summary {
        background: rgba(252,249,242,0.88);
        font-weight: 700;
    }

    /* ── Tabs ────────────────────────────────────────────────────── */
    button[kind="tab"] {
        border-radius: var(--radius-pill);
        border: 1px solid rgba(15,50,42,0.12);
        background: rgba(252,249,242,0.88);
        padding: 0.48rem 1.1rem;
        font-weight: 700;
        color: #224038;
        font-family: "Cabinet Grotesk", system-ui, sans-serif;
        letter-spacing: -0.01em;
        box-shadow: 0 2px 8px rgba(10,30,24,0.05);
        transition: all 0.2s ease;
    }
    button[kind="tab"]:hover:not([aria-selected="true"]) {
        background: rgba(255,255,255,0.92);
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(10,30,24,0.08);
    }
    button[kind="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0f2a24, #0f766e);
        color: #f0ead8 !important;
        border-color: transparent;
        box-shadow: 0 8px 24px rgba(13,118,110,0.25), 0 2px 6px rgba(0,0,0,0.10);
    }

    /* ── Context bar ─────────────────────────────────────────────── */
    .sentree-context {
        margin: 0.6rem 0 1.1rem;
        padding: 0.8rem 1rem;
        border-radius: 16px;
        background: linear-gradient(160deg, rgba(252,249,242,0.96), rgba(239,248,244,0.94));
        border: 1px solid var(--sentree-card-border);
        box-shadow: 0 6px 20px rgba(10,30,24,0.06);
        color: #243b35;
        font-size: 0.88rem;
        font-family: "DM Mono", monospace;
        letter-spacing: 0.01em;
    }

    /* ── Overview grid ───────────────────────────────────────────── */
    .sentree-overview-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.5rem 0 1.1rem;
    }

    .sentree-overview-item {
        border-radius: 18px;
        border: 1px solid var(--sentree-card-border);
        background: rgba(252,249,242,0.90);
        padding: 1rem 1.1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .sentree-overview-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(10,30,24,0.10);
    }
    .sentree-overview-item::before {
        content: attr(data-step);
        position: absolute;
        top: 0.6rem; right: 0.8rem;
        font-size: 2.8rem;
        font-weight: 900;
        font-family: "Fraunces", Georgia, serif;
        color: rgba(13,148,136,0.07);
        line-height: 1;
        pointer-events: none;
    }

    .sentree-overview-item strong {
        display: block;
        margin-bottom: 0.28rem;
        color: #0f2a24;
        font-weight: 800;
        font-size: 0.96rem;
        letter-spacing: -0.01em;
    }

    .sentree-overview-item span {
        color: #3d5850;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Reasoning items ─────────────────────────────────────────── */
    .sentree-reasoning-item {
        margin: 0.2rem 0 0.45rem;
        padding: 0.7rem 0.85rem;
        border-radius: 14px;
        background: rgba(252,249,242,0.70);
        border: 1px solid rgba(13,148,136,0.10);
        border-left: 3px solid var(--sentree-accent);
        transition: background 0.2s ease;
    }
    .sentree-reasoning-item:hover { background: rgba(252,249,242,0.92); }

    /* ── Divider ─────────────────────────────────────────────────── */
    hr { border-color: rgba(15,50,42,0.08); }

    /* ── Dataframe ───────────────────────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--sentree-card-border);
        box-shadow: var(--shadow-card);
    }

    /* ── Caption ─────────────────────────────────────────────────── */
    small, .caption { color: var(--sentree-ink-muted); font-size: 0.82rem; }

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


def _confidence_proxy(entry):
    u_precip = float(entry.get("u_precip", 0.0))
    u_model = float(entry.get("u_model", 0.0))
    u_scenario = float(entry.get("u_scenario", 0.0))
    total_u = min(u_precip + u_model + u_scenario, 0.95)
    return max(0.55, 1.0 - total_u)


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


def _build_investor_rank_table(roi_data_adjusted: dict) -> pd.DataFrame:
    rows = []
    for key, entry in roi_data_adjusted.items():
        meta = INTERVENTIONS.get(key, {})
        roi = float(entry.get("roi", 0.0))
        confidence = float(_confidence_proxy(entry))
        eligible_share = float(entry.get("eligible_share", 0.0))
        tail_nodes = int(entry.get("tail_risk_nodes_neutralized", 0))
        investor_score = roi * confidence * (0.6 + 0.4 * eligible_share) * (1.0 + 0.01 * tail_nodes)

        rows.append(
            {
                "key": key,
                "Intervention": entry.get("name", key.replace("_", " ").title()),
                "Investor Score": investor_score,
                "ROI (x)": roi,
                "Loss Avoided ($M)": float(entry.get("total_loss_avoided", 0.0)) / 1e6,
                "Confidence (%)": confidence * 100.0,
                "Eligible Footprint (%)": eligible_share * 100.0,
                "Tail-Risk Nodes": tail_nodes,
                "Climate Fit": climate_fit_summary(meta) if meta else "broad climate applicability",
            }
        )

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    return ranked.sort_values("Investor Score", ascending=False, ignore_index=True)


def _portfolio_mix(top_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if top_df.empty:
        return top_df
    work = top_df.copy()

    if strategy == "Conservative":
        weight_base = (work["Confidence (%)"] / 100.0) * (work["Eligible Footprint (%)"] / 100.0)
    elif strategy == "Aggressive":
        weight_base = np.power(np.maximum(work["ROI (x)"], 0.0), 1.2)
    else:
        weight_base = np.maximum(work["Investor Score"], 0.0)

    if float(weight_base.sum()) <= 1e-12:
        work["Allocation Weight"] = 1.0 / len(work)
    else:
        work["Allocation Weight"] = weight_base / weight_base.sum()
    return work


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

    BG = "#f6f3ea"
    fig = plt.figure(figsize=(14, 7.4), facecolor=BG)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1],
                            wspace=0.26, hspace=0.34)
    ax_map  = fig.add_subplot(grid[:, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_risk = fig.add_subplot(grid[1, 1])

    for ax in [ax_map, ax_loss, ax_risk]:
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ax.spines.values():
            sp.set_edgecolor((15 / 255, 50 / 255, 42 / 255, 0.12))
        ax.tick_params(colors="#4a6159", labelsize=9)

    if show_edges and training["edge_index"].size > 0:
        edge_index = training["edge_index"]
        segments = [
            [(lons[src], lats[src]), (lons[dst], lats[dst])]
            for src, dst in zip(edge_index[0], edge_index[1])
        ]
        edge_collection = LineCollection(segments, colors=(0.09, 0.22, 0.18, 0.06),
                                         linewidths=0.45)
        ax_map.add_collection(edge_collection)

    scatter = ax_map.scatter(
        lons, lats,
        c=pred, s=12 + pred * 24,
        cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
        alpha=0.88, linewidths=0,
    )

    if highlight_targets:
        tail_mask = target >= training["tail_threshold"]
        ax_map.scatter(
            lons[tail_mask], lats[tail_mask],
            s=46, facecolors="none",
            edgecolors="#0f2a24", linewidths=0.85, alpha=0.72,
        )

    ax_map.set_title(f"Node Risk Field — Epoch {epoch_idx + 1}", loc="left",
                     fontsize=12, fontweight="bold", color="#0f2a24", pad=8)
    ax_map.set_xlabel("Longitude", color="#4a6159", fontsize=10)
    ax_map.set_ylabel("Latitude",  color="#4a6159", fontsize=10)
    ax_map.grid(alpha=0.08, linestyle="--", color="#0f2a24")
    cbar = fig.colorbar(scatter, ax=ax_map, fraction=0.033, pad=0.02)
    cbar.set_label("Predicted systemic risk", color="#4a6159", fontsize=9)
    cbar.ax.tick_params(colors="#4a6159", labelsize=8)

    epochs = training["epochs"]
    ax_loss.plot(epochs, loss, color="#0d9488", linewidth=2.2,
                 solid_capstyle="round")
    ax_loss.fill_between(epochs, loss, alpha=0.08, color="#0d9488")
    ax_loss.scatter([epoch_idx + 1], [loss[epoch_idx]],
                    color="#c2690a", s=52, zorder=3, linewidths=0)
    ax_loss.axvline(epoch_idx + 1, color="#c2690a", linestyle="--",
                    linewidth=1.0, alpha=0.55)
    ax_loss.set_title("Optimization Progress", loc="left", fontsize=11,
                      fontweight="bold", color="#0f2a24", pad=6)
    ax_loss.set_xlabel("Epoch", color="#4a6159", fontsize=9)
    ax_loss.set_ylabel("Huber loss", color="#4a6159", fontsize=9)
    ax_loss.grid(alpha=0.10, linestyle="--", color="#0f2a24")

    ax_risk.plot(epochs, mean_risk, color="#2563eb", linewidth=2.2,
                 label="Mean risk", solid_capstyle="round")
    ax_risk.plot(epochs, p95_risk, color="#b91c1c", linewidth=2.2,
                 label="95th pct", solid_capstyle="round")
    ax_risk.fill_between(epochs, mean_risk, alpha=0.06, color="#2563eb")
    ax_risk.fill_between(epochs, p95_risk, alpha=0.06, color="#b91c1c")
    ax_risk.scatter([epoch_idx + 1], [mean_risk[epoch_idx]],
                    color="#2563eb", s=44, zorder=3, linewidths=0)
    ax_risk.scatter([epoch_idx + 1], [p95_risk[epoch_idx]],
                    color="#b91c1c", s=44, zorder=3, linewidths=0)
    ax_risk.axvline(epoch_idx + 1, color="#c2690a", linestyle="--",
                    linewidth=1.0, alpha=0.55)
    ax_risk.set_title("Prediction Profile", loc="left", fontsize=11,
                      fontweight="bold", color="#0f2a24", pad=6)
    ax_risk.set_xlabel("Epoch", color="#4a6159", fontsize=9)
    ax_risk.set_ylabel("Risk score", color="#4a6159", fontsize=9)
    ax_risk.grid(alpha=0.10, linestyle="--", color="#0f2a24")
    ax_risk.legend(frameon=False, loc="upper left",
                   bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                   fontsize=9, labelcolor="#0f2a24")

    fig.tight_layout(pad=1.6)
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
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    fig.patch.set_facecolor("#faf7f0")
    ax.set_facecolor("#faf7f0")

    palette = {"baseline": "#0f2a24", "intervention": "#0d9488", "fill": "#0d9488"}

    series = [("Baseline", ts["baseline"][metric], palette["baseline"])]
    if intervention_key and intervention_key in ts:
        label = intervention_name or intervention_key.replace("_", " ").title()
        series.append((label, ts[intervention_key][metric], palette["intervention"]))

    for name, values, color in series:
        values_arr = np.array(values)
        ax.plot(years, values_arr, linewidth=2.6, color=color, label=name,
                solid_capstyle="round", solid_joinstyle="round")
        # Shaded band beneath each line
        ax.fill_between(years, values_arr, alpha=0.07, color=color)
        # Terminal dot
        ax.scatter([years[-1]], [values_arr[-1]], color=color, s=52,
                   zorder=4, linewidths=0)

    # Refline at 0.5 (halfway risk)
    ax.axhline(0.5, color="#0d9488", linewidth=0.7, linestyle="--", alpha=0.28)

    ax.set_title("Systemic Risk Trajectory", loc="left", fontsize=13,
                 fontweight="bold", color="#0f2a24", pad=10)
    ax.set_xlabel("Year", color="#4a6159", fontsize=11)
    ax.set_ylabel("P95 RISK" if metric == "p95" else metric.upper(),
                  color="#4a6159", fontsize=11)
    ax.tick_params(colors="#4a6159", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor((15 / 255, 50 / 255, 42 / 255, 0.12))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.10, linestyle="--", color="#0f2a24")
    ax.legend(frameon=False, loc="upper left", ncols=len(series),
              fontsize=11, labelcolor="#0f2a24")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout(pad=1.4)
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


training_history_path = "outputs/roi/gnn_training_history.npz"
training_status = "Training snapshots ready" if os.path.exists(training_history_path) else "Run pipeline to generate playback"
video_count = len([f for f in os.listdir('outputs/videos') if f.endswith('.mp4')]) if os.path.exists('outputs/videos') else 0

logo_markup = ""
logo_path = Path("data/sentree logo.jpg")
if logo_path.exists():
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    logo_markup = (
        "<div class='sentree-hero-logo'>"
        f"<img src='data:image/jpeg;base64,{logo_b64}' alt='SenTree logo' />"
        "</div>"
    )

st.markdown(
    f"""
    <div class="sentree-hero">
        <div class="sentree-hero-grid">
            <div class="sentree-hero-copy">
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
            {logo_markup}
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
        value=int(st.session_state.get("capital_allocation_m", capital_allocation_m)),
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

# Recompute all investment-sensitive outputs from the current slider value.
capital_allocation = int(capital_allocation_m) * 1_000_000
roi_data_adjusted = _apply_capital_allocation(roi_data, capital_allocation)
top_intervention = max(roi_data_adjusted.values(), key=lambda item: item.get("roi", 0.0))
summary_conf = _confidence_proxy(top_intervention)
summary_tail = int(top_intervention.get("tail_risk_nodes_neutralized", 0))
summary_name = top_intervention.get("name", "the selected intervention")
summary_roi = float(top_intervention.get("roi", 0.0))
summary_loss = _format_money_short(top_intervention.get("total_loss_avoided", 0.0))
investor_rank = _build_investor_rank_table(roi_data_adjusted)

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
st.markdown(
    f"""
    <div class="sentree-context">
        <strong>Live Context:</strong> Capital {_format_money_short(capital_allocation)} • Scenario {scenario} • Region SE Asia Coastal •
        Top strategy {top_intervention.get("name", "N/A")} ({top_intervention.get("roi", 0):.2f}x ROI)
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, recommendation_tab, evidence_tab, model_tab = st.tabs(
    ["Overview", "Recommendation", "Evidence", "Model"]
)

with recommendation_tab:
    recommendation_brief_tab, recommendation_compare_tab = st.tabs(["Brief", "Comparison"])

with evidence_tab:
    evidence_search_tab, evidence_videos_tab, evidence_risk_tab, evidence_map_tab = st.tabs(
        ["Search", "Videos", "Risk Over Time", "Map"]
    )

with model_tab:
    model_gnn_tab, model_math_tab = st.tabs(["GNN Playback", "Math Foundations"])

with overview_tab:
    section_header(
        "Overview",
        "Mission snapshot",
        "Use Recommendation for investment decisions, Evidence for media and diagnostics, and Model for training behavior and equations.",
    )
    surface_card(
        "What SenTree does",
        "It predicts tail-risk cascades with a GNN, simulates Koppen-aware interventions, and ranks them by resilience ROI, confidence, and avoided loss.",
    )
    st.markdown(
        """
        <div class="sentree-overview-grid">
            <div class="sentree-overview-item">
                <strong>1) Detect tail-risk</strong>
                <span>The GNN spots nodes entering extreme regimes before losses cascade.</span>
            </div>
            <div class="sentree-overview-item">
                <strong>2) Simulate interventions</strong>
                <span>Koppen-aware intervention rules prevent biome-mismatch recommendations.</span>
            </div>
            <div class="sentree-overview-item">
                <strong>3) Rank investable options</strong>
                <span>Portfolio output combines ROI, confidence, and eligible footprint.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with recommendation_brief_tab:
    section_header(
        "Decide",
        "Investment committee brief",
        "A non-redundant shortlist that blends GNN impact, uncertainty confidence, and Koppen-Geiger climate fit.",
    )
    surface_card(
        "How to read this",
        "Tail-risk nodes are locations where modeled climate risk crosses the extreme (95th percentile) regime, meaning they are most likely to trigger cascading losses. "
        "Investor Score is a blended ranking signal: ROI × confidence × eligible-footprint factor × tail-risk-neutralization bonus.",
    )
    if investor_rank.empty:
        st.info("No intervention ROI results found yet. Run `python scripts/run_pipeline.py` first.")
    else:
        top_shortlist = investor_rank.head(5).copy()
        brief_cols = st.columns([1.4, 1.1])
        with brief_cols[0]:
            st.markdown("**Top interventions (judge-ready shortlist)**")
            st.dataframe(
                top_shortlist[
                    [
                        "Intervention",
                        "Investor Score",
                        "ROI (x)",
                        "Loss Avoided ($M)",
                        "Confidence (%)",
                        "Eligible Footprint (%)",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                height=min(520, 42 * (len(top_shortlist) + 1)),
                column_config={
                    "Investor Score": st.column_config.NumberColumn(format="%.2f"),
                    "ROI (x)": st.column_config.NumberColumn(format="%.2f"),
                    "Loss Avoided ($M)": st.column_config.NumberColumn(format="%.1f"),
                    "Confidence (%)": st.column_config.NumberColumn(format="%.0f"),
                    "Eligible Footprint (%)": st.column_config.NumberColumn(format="%.0f"),
                },
            )
            st.markdown("**Why these 3 interventions?**")
            top_three_reasons = investor_rank.head(3)
            for _, row in top_three_reasons.iterrows():
                st.markdown(
                    f"<div class='sentree-reasoning-item'>"
                    f"<p style='margin:0.02rem 0 0.02rem 0;'>"
                    f"<strong>{row['Intervention']}</strong> | "
                    f"Score {row['Investor Score']:.2f} | "
                    f"Confidence {row['Confidence (%)']:.0f}% | "
                    f"Tail-risk nodes neutralized: {int(row['Tail-Risk Nodes'])}</p>"
                    f"<p style='margin:0; color:#111111; font-weight:400;'>"
                    f"Climate fit: {row['Climate Fit']}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with brief_cols[1]:
            strategy = st.selectbox(
                "Portfolio strategy",
                ["Balanced", "Conservative", "Aggressive"],
                index=0,
                help="Balanced optimizes blended score, Conservative favors confidence and footprint, Aggressive favors higher ROI.",
            )
            top_three = _portfolio_mix(investor_rank.head(3), strategy)
            top_three["Allocated Capital ($M)"] = (
                top_three["Allocation Weight"] * (capital_allocation / 1e6)
            )
            blended_loss = float((top_three["Allocation Weight"] * top_three["Loss Avoided ($M)"]).sum() * 1e6)
            blended_roi = blended_loss / max(float(capital_allocation), 1e-8)
            st.metric("Portfolio ROI", f"{blended_roi:.2f}x")
            st.metric("Portfolio Loss Avoided", _format_money_short(blended_loss))
            mix_chart = (
                alt.Chart(top_three)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Allocated Capital ($M):Q", title="Capital Allocation ($M)"),
                    y=alt.Y("Intervention:N", sort="-x", title=""),
                    color=alt.Color("Intervention:N", legend=None, scale=alt.Scale(scheme="teals")),
                    tooltip=[
                        "Intervention",
                        alt.Tooltip("Allocated Capital ($M):Q", format=".1f"),
                        alt.Tooltip("ROI (x):Q", format=".2f"),
                        alt.Tooltip("Confidence (%):Q", format=".0f"),
                    ],
                )
                .properties(height=190)
            )
            st.altair_chart(mix_chart, use_container_width=True)

            st.markdown("**Investment memo lines**")
            for _, row in top_three.iterrows():
                st.markdown(
                    f"- Allocate about {_format_money_short(row['Allocated Capital ($M)'] * 1e6)} to **{row['Intervention']}** "
                    f"(ROI {row['ROI (x)']:.2f}x, confidence {row['Confidence (%)']:.0f}%)."
                )

with evidence_search_tab:
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

with evidence_search_tab:
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
with recommendation_compare_tab:
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

with model_gnn_tab:
    section_header(
        "Playback",
        "GNN training playback",
        "View the dedicated React playback UI directly inside Streamlit, with a Streamlit-native fallback for quick checks.",
    )

    playback_mode = st.radio(
        "Playback UI",
        ["Embedded React app", "Streamlit fallback"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="sentree_playback_mode",
    )

    if playback_mode == "Embedded React app":
        import streamlit.components.v1 as components

        default_url = os.environ.get("SENTREE_GNN_PLAYBACK_URL", "http://localhost:4173/").strip()
        playback_url = st.text_input(
            "React app URL",
            value=default_url,
            help=(
                "This should be reachable from your laptop browser. "
                "When using SSH port forwarding, this is typically http://localhost:4173/."
            ),
            key="sentree_gnn_playback_url",
        )
        st.caption(
            "If the embed is blank, verify the Vite dev server is running on the compute node and your SSH tunnel forwards its port."
        )
        components.iframe(playback_url, height=920, scrolling=True)
    else:
        training = load_training_history()
        if training is None:
            st.info(
                "Training history not found yet. Re-run `python scripts/run_pipeline.py` to generate `outputs/roi/gnn_training_history.npz`."
            )
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

with evidence_tab:
    ts = load_risk_timeseries()
    opportunity = load_opportunity_map()

with evidence_videos_tab:
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
                p for p in videos
                if p.stem.startswith("interventions_") or p.stem.startswith("megavideo_") or "grid" in p.stem or "megavideo" in p.stem
            ]
            core_videos = [p for p in videos if p.stem in {"baseline_risk", "tail_risk_escalation", "climate_classification_shift"}]

            video_type = st.selectbox(
                "Video type",
                ["Comparison", "Core Maps", "Grid"],
                index=0,
                key="evidence_video_type",
                help="Use this dropdown instead of tabs to avoid horizontal scrolling when many videos exist.",
            )
            if video_type == "Comparison":
                if not comparison_videos:
                    st.info("No comparison videos found yet. Render them via `bash scripts/submit_render_comparisons.sh`.")
                else:
                    def _cmp_label(p: Path) -> str:
                        key = p.stem[len("comparison_"):] if p.stem.startswith("comparison_") else p.stem
                        name = INTERVENTIONS.get(key, {}).get("name") or key.replace("_", " ").title()
                        return f"{name}"

                    options = {_cmp_label(p): p for p in sorted(comparison_videos, key=_cmp_label)}
                    label = st.selectbox("Select intervention", list(options.keys()), index=0, key="evidence_video_cmp")
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
                    label = st.selectbox("Select map", list(options.keys()), index=0, key="evidence_video_core")
                    st.caption(str(options[label]))
                    _show_video(str(options[label]))
            else:
                if not grid_videos:
                    st.info(
                        "No grid/mega videos found yet.\n\n"
                        "Render one with:\n"
                        "`python scripts/render_megavideo_from_npz.py --mode grid --out outputs/videos/interventions_grid.mp4 --ncols 6`"
                    )
                else:
                    options = {p.stem.replace("_", " ").title(): p for p in grid_videos}
                    label = st.selectbox("Select grid video", list(options.keys()), index=0, key="evidence_video_grid")
                    st.caption(str(options[label]))
                    _show_video(str(options[label]))
        else:
            st.info("No videos generated yet. Run `python scripts/run_pipeline.py`.")
    else:
        st.info("Output directory not found. Run the pipeline first.")

with evidence_risk_tab:
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
        category_choice = st.selectbox("Category", category_options, index=0, key="evidence_risk_category")

        scoped_keys = category_map.get(category_choice, [])
        scoped_labels = ["Baseline"]
        scoped_lookup = {"Baseline": None}
        for key in scoped_keys:
            label = INTERVENTIONS.get(key, {}).get("name", key.replace("_", " ").title())
            scoped_labels.append(label)
            scoped_lookup[label] = key

        choice = st.selectbox("Intervention", scoped_labels, index=0, key="evidence_risk_intervention")
        chosen_key = scoped_lookup.get(choice)
        fig = build_risk_timeseries_figure(ts, "p95", chosen_key, choice if chosen_key else None)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with evidence_map_tab:
    section_header(
        "Locate",
        "Tail-risk escalation map",
        "Scan the geography of exposure and opportunity. Red overlays indicate nodes exceeding the model's extreme-regime threshold.",
    )
    if opportunity is not None:
        with st.expander("Interactive map (optional)", expanded=False):
            st.caption("Optional 2D/3D basemap view. The default below is the static ROI PNG.")
            map_mode = st.selectbox(
                "Interactive map mode",
                ["2D ROI map (basemap)", "3D ROI extrusion (slower)"],
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
                tooltip = {"text": "Avoided damage potential: {value}\nNearest: {nearest_city} (~{nearest_km} km)"}
                deck = pdk.Deck(
                    layers=[layer_cells],
                    initial_view_state=view,
                    tooltip=tooltip,
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                )
                st.pydeck_chart(deck, width="stretch")
                st.caption(f"Color scale uses min/max of the opportunity grid: vmin={vmin:.4f}, vmax={vmax:.4f}.")
            except Exception as e:
                st.info(f"Interactive map unavailable ({e}).")

    tail_risk_img = "outputs/tail_risk_map.png"
    if os.path.exists(tail_risk_img):
        st.image(tail_risk_img, width="stretch")
    else:
        st.info("Tail-risk map not generated yet.")

with model_math_tab:
    render_math_view()

# --- Footer ---
st.markdown("""
<div style="
    margin-top:2.5rem;
    padding:1.1rem 1.4rem;
    border-radius:18px;
    background:linear-gradient(135deg,rgba(15,42,36,0.06),rgba(13,148,136,0.06));
    border:1px solid rgba(15,50,42,0.08);
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    font-size:0.82rem;
    color:#4a6159;
    font-family:'DM Mono',monospace;
    letter-spacing:0.01em;
">
    <span>SenTree — Resilience ROI Dashboard</span>
    <span style="color:rgba(74,97,89,0.5)">|</span>
    <span>ML@Purdue Catapult Hackathon</span>
    <span style="color:rgba(74,97,89,0.5)">|</span>
    <span>SE Asia · SSP3-7.0 · 2015–2100</span>
</div>
""", unsafe_allow_html=True)
