"""
Back-compat embedding entry points.

Historically this repo exposed `src.embedding.embedder.embed_video(...)`.
Keep that import path stable, but route calls through the new embedder
factory in `src.embedding`.
"""

from __future__ import annotations

import numpy as np

from . import embed_video as _embed_video


def embed_video(video_path, metadata=None, n_frames=8, use_gemini=True):
    """
    Embed a video as a single vector (dims=768 by default).

    Notes:
    - `n_frames` is ignored by the Gemini Embeddings 2 backend.
    - Video indexing remains native-video only; `metadata` is kept only for
      API compatibility and is not mixed into Gemini document embeddings.
    - `use_gemini=False` uses the local fallback embedder.
    """
    backend = "gemini" if use_gemini else "local"
    emb = _embed_video(video_path, metadata=metadata, backend=backend)
    emb = np.asarray(emb, dtype=np.float32)
    if emb.size == 0:
        return np.zeros(768, dtype=np.float32)
    # Normalize for cosine distance search.
    return emb / (np.linalg.norm(emb) + 1e-8)

