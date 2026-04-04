"""
Embedder factory — selects and caches the active backend.

Provides backward-compatible convenience functions (embed_video, embed_query).
Re-exports error classes from gemini_embedder for existing import sites.
"""

from __future__ import annotations

from .base_embedder import BaseEmbedder
from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError  # noqa: F401

_current_embedder: BaseEmbedder | None = None
_current_backend: str | None = None


def get_embedder(*, backend: str = "gemini", **kwargs) -> BaseEmbedder:
    """Factory to get or create the active embedder."""
    global _current_embedder, _current_backend

    if _current_embedder is not None and _current_backend == backend:
        return _current_embedder

    if backend == "gemini":
        from .gemini_embedder import GeminiEmbedder

        _current_embedder = GeminiEmbedder(
            rpm=kwargs.get("rpm", None),
            model=kwargs.get("model", None),
            dimensions=kwargs.get("dimensions", None),
        )
    elif backend == "local":
        from .local_embedder import LocalEmbedder

        _current_embedder = LocalEmbedder(dimensions=kwargs.get("dimensions", 768))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    _current_backend = backend
    return _current_embedder


def reset_embedder() -> None:
    """Reset the cached embedder (for switching backends)."""
    global _current_embedder, _current_backend
    _current_embedder = None
    _current_backend = None


def embed_video(video_path: str, *, metadata: str | None = None, backend: str = "gemini", verbose: bool = False) -> list[float]:
    return get_embedder(backend=backend).embed_video(video_path, metadata=metadata, verbose=verbose)


def embed_query(query_text: str, *, backend: str = "gemini", verbose: bool = False) -> list[float]:
    return get_embedder(backend=backend).embed_query(query_text, verbose=verbose)
