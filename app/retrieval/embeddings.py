from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Thin wrapper around sentence-transformers for document/query embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vector[0].tolist()
