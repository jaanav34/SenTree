"""ChromaDB vector store for video search."""
import chromadb
import numpy as np
import os
from pathlib import Path
from datetime import datetime

from src.embedding import embed_query
from src.embedding.gemini_embedder import GeminiAPIKeyError


def _looks_like_corrupt_hnsw_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("hnsw segment reader" in msg) or ("nothing found on disk" in msg)


class VideoSearchDB:
    def __init__(self, persist_dir='outputs/embeddings'):
        # Resolve persistent path relative to repo root so it works no matter the CWD
        # (e.g. Streamlit often runs from a different working directory).
        repo_root = Path(__file__).resolve().parents[2]
        persist_dir = os.environ.get("SENTREE_CHROMA_DIR", persist_dir)
        persist_path = Path(persist_dir)
        if not persist_path.is_absolute():
            persist_path = repo_root / persist_path

        self.persist_path = persist_path
        os.makedirs(persist_path, exist_ok=True)
        self._open()

    def _open(self) -> None:
        """Open (or recover) the Chroma persistent collection."""
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_path))
            self.collection = self.client.get_or_create_collection(
                name='sentree_videos',
                metadata={'hnsw:space': 'cosine'}
            )
        except Exception as e:
            if not _looks_like_corrupt_hnsw_error(e):
                raise

            # Corrupt on-disk HNSW index (common after interrupted writes on Windows).
            # Move aside and recreate a fresh store.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = self.persist_path.with_name(f"{self.persist_path.name}_corrupt_{ts}")
            try:
                if self.persist_path.exists() and not backup.exists():
                    self.persist_path.rename(backup)
            except Exception:
                # If rename fails (e.g. locked files), just re-raise the original error.
                raise

            os.makedirs(self.persist_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_path))
            self.collection = self.client.get_or_create_collection(
                name='sentree_videos',
                metadata={'hnsw:space': 'cosine'}
            )

    def add_video(self, video_id, embedding, metadata=None):
        """Add a video embedding to the store."""
        try:
            self.collection.upsert(
                ids=[video_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata or {}],
            )
        except Exception as e:
            if _looks_like_corrupt_hnsw_error(e):
                self._open()
                self.collection.upsert(
                    ids=[video_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata or {}],
                )
                return
            raise

    def add_videos_batch(self, video_ids, embeddings, metadatas=None):
        """Batch add videos."""
        self.collection.upsert(
            ids=video_ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas or [{} for _ in video_ids],
        )

    def query(self, query_text, n_results=5, use_gemini=True):
        """Search for videos matching a text query in Gemini's shared embedding space."""
        query_embedding = self._embed_query(query_text, use_gemini)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
            )
        except Exception as e:
            if _looks_like_corrupt_hnsw_error(e):
                self._open()
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                )
            else:
                raise
        return results

    def _embed_query(self, text, use_gemini=True):
        """Embed query text using the same Gemini model family as indexed videos."""
        if not use_gemini:
            raise ValueError("SenTree search is configured for Gemini embeddings only.")

        try:
            emb = np.array(embed_query(text, backend="gemini"), dtype=np.float32)
        except GeminiAPIKeyError:
            raise
        except Exception as e:
            raise RuntimeError(f"Gemini query embedding failed: {e}") from e

        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm

    def count(self):
        return self.collection.count()
