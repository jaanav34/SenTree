from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, TypeVar


DIMENSIONS = int(os.environ.get("SENTREE_EMBED_DIMENSIONS", "768"))
DEFAULT_RPM = int(os.environ.get("SENTREE_GEMINI_RPM", "55"))

# Gemini Embeddings 2 (preview) per user request; can be overridden.
EMBED_MODEL = os.environ.get("SENTREE_GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")


class GeminiAPIKeyError(RuntimeError):
    """Raised when Gemini API key is missing."""


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini API quota/rate-limit is exceeded."""


T = TypeVar("T")


def _load_dotenv_if_present(dotenv_path: Path) -> None:
    """
    Minimal .env loader.

    - Supports KEY=VALUE lines (optionally quoted).
    - Ignores blank lines and comments starting with '#'.
    - Does not overwrite already-set environment variables.
    """
    try:
        if not dotenv_path.exists():
            return
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return


def _get_api_key() -> str | None:
    # Common env var names used by different SDKs/docs.
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    repo_root = Path(__file__).resolve().parents[2]
    _load_dotenv_if_present(repo_root / ".env")
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


class _RateLimiter:
    """Simple sliding-window rate limiter based on request timestamps."""

    def __init__(self, max_per_minute: int = DEFAULT_RPM):
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


def _retry(fn: Callable[[], T], *, max_retries: int = 5, initial_delay: float = 2.0, max_delay: float = 60.0) -> T:
    """Call *fn* with exponential back-off on transient errors (429, 503)."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            retryable = status in (429, 503) or ("resource exhausted" in msg) or ("429" in msg) or ("503" in msg)
            if not retryable or attempt == max_retries:
                if ("resource exhausted" in msg) or (status == 429):
                    raise GeminiQuotaError(
                        "Gemini API quota/rate limit exceeded.\n\n"
                        "Options:\n"
                        "  - Wait and retry\n"
                        "  - Reduce indexing volume (fewer videos)\n"
                        "  - Upgrade/enable billing for your API key"
                    ) from exc
                raise

            wait_for = min(delay, max_delay)
            print(
                f"  Retryable Gemini error (attempt {attempt + 1}/{max_retries}), waiting {wait_for:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait_for)
            delay *= 2


def _make_video_part(video_path: str, types):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    if hasattr(types.Part, "from_bytes"):
        return types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    return types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"))


class GeminiEmbedder:
    """Gemini Embeddings 2 backend (API-based)."""

    def __init__(
        self,
        *,
        rpm: int | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        from google import genai

        api_key = _get_api_key()
        if not api_key:
            raise GeminiAPIKeyError(
                "Gemini API key not found.\n\n"
                "Set one of these (or add to `.env`):\n"
                "  - GOOGLE_API_KEY=...\n"
                "  - GEMINI_API_KEY=...\n"
            )

        rpm = DEFAULT_RPM if rpm is None else int(rpm)
        model = EMBED_MODEL if model is None else str(model)
        dimensions = DIMENSIONS if dimensions is None else int(dimensions)

        self._client = genai.Client(api_key=api_key)
        self._limiter = _RateLimiter(max_per_minute=rpm)
        self._model = model
        self._dimensions = dimensions

    def embed_video(self, video_path: str, *, metadata: str | None = None, verbose: bool = False) -> list[float]:
        from google.genai import types

        video_part = _make_video_part(video_path, types)
        parts = [video_part]
        if metadata:
            parts.append(types.Part.from_text(text=str(metadata)))

        if verbose:
            size_kb = os.path.getsize(video_path) / 1024
            print(f"    [verbose] sending {size_kb:.0f}KB to {self._model}", file=sys.stderr)

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.embed_content(
                model=self._model,
                contents=types.Content(parts=parts),
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._dimensions,
                ),
            )
        )
        elapsed = time.monotonic() - t0

        embedding = response.embeddings[0].values
        if verbose:
            print(f"    [verbose] dims={len(embedding)}, api_time={elapsed:.2f}s", file=sys.stderr)
        return embedding

    def embed_query(self, query_text: str, *, verbose: bool = False) -> list[float]:
        from google.genai import types

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.embed_content(
                model=self._model,
                contents=query_text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self._dimensions,
                ),
            )
        )
        elapsed = time.monotonic() - t0

        embedding = response.embeddings[0].values
        if verbose:
            print(f"  [verbose] query dims={len(embedding)}, api_time={elapsed:.2f}s", file=sys.stderr)
        return embedding

    def dimensions(self) -> int:
        return self._dimensions
