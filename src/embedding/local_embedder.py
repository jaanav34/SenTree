from __future__ import annotations

import numpy as np

from .base_embedder import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """
    Local fallback embedder.

    Currently uses CLIP if installed; otherwise returns zeros.
    """

    def __init__(self, *, dimensions: int = 768):
        self._dimensions = dimensions

    def embed_video(self, video_path: str, *, metadata: str | None = None, verbose: bool = False) -> list[float]:
        # No robust local video embedding available here; return zeros to keep pipeline moving.
        return np.zeros(self._dimensions, dtype=np.float32).tolist()

    def embed_query(self, query_text: str, *, verbose: bool = False) -> list[float]:
        try:
            import open_clip
            import torch

            model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model.eval()
            with torch.no_grad():
                tokens = tokenizer([query_text])
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.squeeze().cpu().numpy().astype(np.float32).tolist()
        except Exception:
            return np.zeros(self._dimensions, dtype=np.float32).tolist()

    def dimensions(self) -> int:
        return self._dimensions

