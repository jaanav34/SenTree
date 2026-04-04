"""ChromaDB vector store for video search."""
import chromadb
import numpy as np
import os


class VideoSearchDB:
    def __init__(self, persist_dir='outputs/embeddings'):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name='sentree_videos',
            metadata={'hnsw:space': 'cosine'}
        )

    def add_video(self, video_id, embedding, metadata=None):
        """Add a video embedding to the store."""
        self.collection.upsert(
            ids=[video_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata or {}],
        )

    def add_videos_batch(self, video_ids, embeddings, metadatas=None):
        """Batch add videos."""
        self.collection.upsert(
            ids=video_ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas or [{} for _ in video_ids],
        )

    def query(self, query_text, n_results=5, use_gemini=True):
        """Search for videos matching a text query."""
        query_embedding = self._embed_text(query_text, use_gemini)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )
        return results

    def _embed_text(self, text, use_gemini=True):
        """Embed query text."""
        if use_gemini:
            try:
                from google import genai
                client = genai.Client()
                response = client.models.embed_content(
                    model='models/text-embedding-004',
                    contents=text
                )
                return np.array(response.embeddings[0].values)
            except Exception as e:
                print(f"Gemini text embed failed: {e}")

        # Fallback: CLIP text embedding
        try:
            import open_clip
            import torch
            model, _, _ = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k'
            )
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            with torch.no_grad():
                tokens = tokenizer([text])
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.squeeze().numpy()
        except Exception:
            print("WARNING: No embedding model available. Returning zeros.")
            return np.zeros(768)

    def count(self):
        return self.collection.count()
