from __future__ import annotations

from app.models.schemas import RetrievedChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


class SemanticRetriever:
    """High-level semantic search utility used by the agent retrieval tool."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 6) -> list[RetrievedChunk]:
        query_embedding = self.embedding_service.embed_query(query)
        return self.vector_store.query(query_embedding=query_embedding, top_k=top_k)
