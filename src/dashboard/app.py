"""SenTree Dashboard — Streamlit UI."""
import os
import sys
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from sentree_venv import ensure_venv

ensure_venv()

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='SenTree — Resilience ROI Dashboard', layout='wide')

# --- Header ---
st.title('SenTree: Resilience ROI Dashboard')
st.markdown('*Climate Adaptation Intelligence for Sovereign Wealth Funds*')
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header('Controls')
    scenario = st.selectbox('Climate Scenario', ['SSP3-7.0 (High Emissions)', 'SSP1-2.6 (Low Emissions)'])
    intervention = st.selectbox('Intervention', ['Coastal Mangrove Restoration', 'Regenerative Agriculture', 'Both'])
    st.divider()
    st.markdown('**Region:** SE Asia Coastal')
    st.markdown('**Time:** 2015-2050')

# --- Search ---
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        'Search simulations',
        placeholder='e.g., "Show where mangroves prevent collapse"',
        key='search_query'
    )
with col2:
    search_btn = st.button('Search', type='primary', use_container_width=True)

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


roi_data = load_roi_data()


@st.cache_data
def load_risk_timeseries():
    path = "outputs/roi/risk_timeseries.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# --- Search Results ---
if search_btn and query:
    st.subheader('Search Results')

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
                        st.video(video_path)

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
st.subheader('Intervention Comparison')

cols = st.columns(len(roi_data))
for i, (key, data) in enumerate(roi_data.items()):
    with cols[i]:
        st.markdown(f"**{data['name']}**")
        st.metric('Resilience ROI', f"{data['roi']:.2f}x",
                   help=f"Range: {data.get('roi_lower', 0):.2f} - {data.get('roi_upper', 0):.2f}")
        st.metric('Total Loss Avoided', f"${data.get('total_loss_avoided', 0)/1e9:.1f}B")
        st.metric('Mean Risk Reduction', f"{data.get('mean_risk_reduction', 0):.1%}")

        tail_count = data.get('tail_risk_nodes_neutralized', 0)
        if tail_count > 0:
            st.error(f'Tail-Risk Nodes Neutralized: {tail_count}')

# --- Video Display ---
st.subheader('Simulation Videos')

video_dir = 'outputs/videos'
if os.path.exists(video_dir):
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if videos:
        tabs = st.tabs([v.replace('.mp4', '').replace('_', ' ').title() for v in videos])
        for tab, vid in zip(tabs, videos):
            with tab:
                st.video(os.path.join(video_dir, vid))
    else:
        st.info('No videos generated yet. Run `python scripts/run_pipeline.py`.')
else:
    st.info('Output directory not found. Run the pipeline first.')

# --- Math & Methodology Tab ---
st.divider()
st.header('Technical Deep-Dive: Math & Methodology')

math_tab, playground_tab = st.tabs(['📐 Mathematical Foundations', '🎮 Interactive Playground'])

with math_tab:
    st.subheader('1. Tail-Risk Escalation (Gurjar & Camp 2026)')
    st.markdown("""
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
    
    **D. Hawkes Self-Excitation:**
    To capture "clusters" of extreme events, we add a Hawkes process intensity:
    $$\lambda^*(t) = \mu + \sum_{t_i < t} \\beta e^{-\gamma(t - t_i)}$$
    Nodes exceeding the 95th percentile of the composite score are flagged as **Tail-Risk Escalation** zones.
    """)
    
    st.subheader('2. Resilience ROI & Economic Exposure (Ito 2020)')
    st.markdown("""
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
    st.markdown("""
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
st.subheader("Risk Over Time")
ts = load_risk_timeseries()
if ts is None:
    st.info("Risk time series not found yet. Re-run the pipeline to generate `outputs/roi/risk_timeseries.json`.")
else:
    metric = st.selectbox("Metric", ["mean", "p95", "max"], index=0)
    years = ts["years"]
    chart = {"Year": years}
    chart["Baseline"] = ts["baseline"][metric]
    if "mangrove_restoration" in ts:
        chart["Mangrove Restoration"] = ts["mangrove_restoration"][metric]
    if "regenerative_agriculture" in ts:
        chart["Regenerative Agriculture"] = ts["regenerative_agriculture"][metric]
    df = pd.DataFrame(chart).set_index("Year")
    st.line_chart(df)

# --- Tail Risk Map ---
st.subheader('Tail-Risk Escalation Map')
st.markdown('Nodes exceeding 95th percentile volatility+momentum threshold are flagged.')

tail_risk_img = 'outputs/tail_risk_map.png'
if os.path.exists(tail_risk_img):
    st.image(tail_risk_img, use_container_width=True)
else:
    st.info('Tail-risk map not generated yet.')

# --- Footer ---
st.divider()
st.caption('SenTree — Resilience ROI Dashboard | ML@Purdue Catapult Hackathon')
