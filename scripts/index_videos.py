"""Embed all output videos into ChromaDB for semantic search."""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.embedder import embed_video
from src.embedding.vectordb import VideoSearchDB

VIDEO_DIR = 'outputs/videos'
ROI_PATH = 'outputs/roi/roi_results.json'

roi_data = {}
if os.path.exists(ROI_PATH):
    with open(ROI_PATH, 'r') as f:
        roi_data = json.load(f)

VIDEO_META = {
    'baseline_risk': {
        'title': 'Baseline Climate Risk — SSP3-7.0 SE Asia',
        'description': 'Baseline systemic climate risk across Southeast Asia coastal region under SSP3-7.0 scenario, showing temperature and precipitation-driven risk propagation',
        'intervention_key': '',
        'has_tail_risk': True,
    },
    'tail_risk_escalation': {
        'title': 'Tail-Risk Escalation Events — Tipping Points',
        'description': 'Nodes exceeding 95th percentile volatility and momentum thresholds, indicating potential ecosystem collapse tipping points',
        'intervention_key': '',
        'has_tail_risk': True,
    },
    'comparison_mangrove_restoration': {
        'title': 'Mangrove Restoration Impact — Before vs After',
        'description': 'Side-by-side comparison showing how coastal mangrove restoration reduces systemic risk, prevents collapse in coastal nodes',
        'intervention_key': 'mangrove_restoration',
        'has_tail_risk': True,
    },
    'mangrove_restoration_risk': {
        'title': 'Risk with Mangrove Restoration',
        'description': 'Climate risk map after applying coastal mangrove restoration intervention, showing reduced coastal vulnerability',
        'intervention_key': 'mangrove_restoration',
        'has_tail_risk': False,
    },
    'comparison_regenerative_agriculture': {
        'title': 'Regenerative Agriculture Impact — Before vs After',
        'description': 'Side-by-side comparison showing how regenerative agriculture reduces precipitation volatility and stabilizes agricultural regions',
        'intervention_key': 'regenerative_agriculture',
        'has_tail_risk': True,
    },
    'regenerative_agriculture_risk': {
        'title': 'Risk with Regenerative Agriculture',
        'description': 'Climate risk map after applying regenerative agriculture intervention across the region',
        'intervention_key': 'regenerative_agriculture',
        'has_tail_risk': False,
    },
}

print("Indexing videos into ChromaDB...")
db = VideoSearchDB()

videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
for vid_file in videos:
    vid_id = vid_file.replace('.mp4', '')
    vid_path = os.path.join(VIDEO_DIR, vid_file)
    meta = VIDEO_META.get(vid_id, {
        'title': vid_id.replace('_', ' ').title(),
        'description': f'Climate simulation video: {vid_id}',
    })

    ikey = meta.get('intervention_key', '')
    if ikey and ikey in roi_data:
        meta['roi'] = roi_data[ikey].get('roi', 0)
        meta['tail_risk_count'] = roi_data[ikey].get('tail_risk_nodes_neutralized', 0)

    meta['video_path'] = vid_path

    print(f"  Embedding: {vid_file}...")
    try:
        embedding = embed_video(vid_path, metadata=meta.get('description'), use_gemini=True)
        db.add_video(vid_id, embedding, metadata=meta)
        print(f"    Done")
    except Exception as e:
        print(f"    Failed: {e}")

print(f"\nDone. {db.count()} videos indexed.")
