"""Embed video frames using Gemini API or CLIP fallback."""
import os
import numpy as np
from PIL import Image
import tempfile


def extract_keyframes(video_path, n_frames=8):
    """Extract evenly-spaced frames from MP4."""
    import subprocess
    import json

    probe = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path],
        capture_output=True, text=True
    )
    info = json.loads(probe.stdout)
    duration = float(info['streams'][0].get('duration', '10'))

    tmpdir = tempfile.mkdtemp()
    frames = []
    timestamps = np.linspace(0, duration * 0.95, n_frames)

    for i, ts in enumerate(timestamps):
        out_path = os.path.join(tmpdir, f'frame_{i:03d}.png')
        subprocess.run([
            'ffmpeg', '-ss', str(ts), '-i', video_path,
            '-frames:v', '1', '-q:v', '2', out_path,
            '-y', '-loglevel', 'quiet'
        ])
        if os.path.exists(out_path):
            frames.append(out_path)

    return frames


def embed_with_gemini(frame_paths, video_metadata=None):
    """Embed video frames using Gemini multimodal -> text description -> text embedding."""
    from google import genai

    client = genai.Client()

    embeddings = []
    for path in frame_paths:
        img = Image.open(path)

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                img,
                'Describe this climate risk heatmap in detail: regions, risk levels, patterns, notable features. Be specific about geography and severity.'
            ]
        )
        description = response.text

        if video_metadata:
            description = f"{video_metadata}. {description}"

        embed_response = client.models.embed_content(
            model='models/text-embedding-004',
            contents=description
        )
        embeddings.append(embed_response.embeddings[0].values)

    return embeddings


def embed_with_clip_fallback(frame_paths):
    """Fallback: use CLIP to embed frames directly."""
    try:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        model.eval()

        embeddings = []
        for path in frame_paths:
            img = preprocess(Image.open(path)).unsqueeze(0)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze().numpy())

        return embeddings
    except ImportError:
        print("CLIP not installed. Install with: pip install open-clip-torch")
        return None


def embed_video(video_path, metadata=None, n_frames=8, use_gemini=True):
    """Main entry point: embed a video, return mean embedding."""
    frames = extract_keyframes(video_path, n_frames=n_frames)

    if not frames:
        print(f"WARNING: No frames extracted from {video_path}")
        return np.zeros(768)

    if use_gemini:
        try:
            embeddings = embed_with_gemini(frames, metadata)
        except Exception as e:
            print(f"Gemini failed: {e}. Falling back to CLIP.")
            embeddings = embed_with_clip_fallback(frames)
    else:
        embeddings = embed_with_clip_fallback(frames)

    if embeddings is None or len(embeddings) == 0:
        return np.zeros(768)

    mean_emb = np.mean(embeddings, axis=0)
    return mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
