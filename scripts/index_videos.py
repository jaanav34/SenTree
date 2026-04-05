"""Embed output videos into ChromaDB for semantic search."""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentree_venv import ensure_venv

ensure_venv()

from src.embedding.embedder import embed_video
from src.embedding.vectordb import VideoSearchDB
from src.simulation.interventions import INTERVENTIONS, build_search_description, climate_fit_summary

VIDEO_DIR = 'outputs/videos'
ROI_PATH = 'outputs/roi/roi_results.json'


def _load_roi_data():
    if os.path.exists(ROI_PATH):
        with open(ROI_PATH, 'r') as f:
            return json.load(f)
    return {}


def _base_video_meta():
    return {
        'baseline_risk': {
            'title': 'Baseline Climate Risk',
            'description': (
                'Baseline systemic climate risk map showing temperature and precipitation-driven '
                'cascade behavior before any adaptation action is applied.'
            ),
            'intervention_key': '',
            'has_tail_risk': True,
            'category': 'baseline',
        },
        'tail_risk_escalation': {
            'title': 'Tail-Risk Escalation Events',
            'description': (
                'Tail-risk escalation map highlighting nodes above the 95th percentile for '
                'volatility and momentum, signaling potential tipping points.'
            ),
            'intervention_key': '',
            'has_tail_risk': True,
            'category': 'tail_risk',
        },
        'climate_classification_shift': {
            'title': 'Koppen-Geiger Climate Classification Shifts',
            'description': (
                'Koppen-Geiger climate zones evolving over time, useful for matching interventions '
                'to climate suitability and understanding biome transitions.'
            ),
            'intervention_key': '',
            'has_tail_risk': False,
            'category': 'climate_classification',
        },
    }


def _intervention_video_meta():
    meta = {}
    for key, intervention in INTERVENTIONS.items():
        description = build_search_description(key, intervention)
        meta[f'comparison_{key}'] = {
            'title': f"{intervention['name']} Impact Comparison",
            'description': (
                f"Before-vs-after intervention comparison. {description}"
            ),
            'intervention_key': key,
            'has_tail_risk': True,
            'category': intervention.get('category', 'intervention'),
            'climate_fit': climate_fit_summary(intervention),
            'search_tags': ", ".join(intervention.get('search_tags', [])),
        }
    return meta


print("Indexing videos into ChromaDB...")
db = VideoSearchDB()
roi_data = _load_roi_data()
video_meta = _base_video_meta()
video_meta.update(_intervention_video_meta())

videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
for vid_file in videos:
    vid_id = vid_file.replace('.mp4', '')
    vid_path = os.path.join(VIDEO_DIR, vid_file)
    meta = dict(video_meta.get(vid_id, {
        'title': vid_id.replace('_', ' ').title(),
        'description': f'Climate simulation video: {vid_id}',
        'category': 'simulation',
    }))

    ikey = meta.get('intervention_key', '')
    if ikey and ikey in roi_data:
        roi_entry = roi_data[ikey]
        meta['roi'] = roi_entry.get('roi', 0)
        meta['tail_risk_count'] = roi_entry.get('tail_risk_nodes_neutralized', 0)
        meta['eligible_nodes'] = roi_entry.get('eligible_nodes', 0)
        meta['eligible_share'] = roi_entry.get('eligible_share', 0.0)

    meta['video_path'] = vid_path

    print(f"  Embedding: {vid_file}...")
    try:
        embedding = embed_video(vid_path, metadata=meta.get('description'), use_gemini=True)
        db.add_video(vid_id, embedding, metadata=meta)
        print("    Done")
    except Exception as e:
        print(f"    Failed: {e}")

print(f"\nDone. {db.count()} videos indexed.")