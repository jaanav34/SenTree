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
