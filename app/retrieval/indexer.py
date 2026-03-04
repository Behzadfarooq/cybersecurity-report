from __future__ import annotations

import logging

from app.models.schemas import ChunkRecord
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)


def index_chunks(
    chunks: list[ChunkRecord],
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    batch_size: int = 64,
) -> None:
    """Generate embeddings and upsert chunks into vector database in batches."""

    if not chunks:
        LOGGER.warning("No chunks provided for indexing.")
        return

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.content for chunk in batch]
        embeddings = embedding_service.embed_texts(texts=texts)
        vector_store.upsert_chunks(chunks=batch, embeddings=embeddings)

        LOGGER.info(
            "Indexed chunk batch",
            extra={
                "batch_start": start,
                "batch_size": len(batch),
                "total": len(chunks),
            },
        )
