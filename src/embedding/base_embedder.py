from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_video(self, video_path: str, *, metadata: str | None = None, verbose: bool = False) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query_text: str, *, verbose: bool = False) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

