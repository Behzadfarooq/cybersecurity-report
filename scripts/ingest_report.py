from __future__ import annotations

import argparse
import logging

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.etl.pipeline import ETLPipeline
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.indexer import index_chunks
from app.retrieval.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Cyber Ireland 2022 PDF and build vector index.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the PDF again even if it already exists locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    LOGGER.info("Starting ingestion pipeline")

    etl = ETLPipeline(settings=settings)
    chunks = etl.run(force_download=args.force_download)

    embedding_service = EmbeddingService(model_name=settings.embedding_model)
    vector_store = VectorStore(
        persist_path=settings.vector_db_absolute_path,
        collection_name=settings.collection_name,
    )
    index_chunks(chunks=chunks, embedding_service=embedding_service, vector_store=vector_store)

    LOGGER.info(
        "Ingestion complete",
        extra={
            "chunks": len(chunks),
            "indexed_chunks": vector_store.count(),
            "chunk_file": str(settings.chunks_absolute_path),
            "vector_db": str(settings.vector_db_absolute_path),
        },
    )


if __name__ == "__main__":
    main()
