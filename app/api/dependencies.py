from __future__ import annotations

from functools import lru_cache

from app.agents.query_agent import QueryAgent
from app.config.settings import get_settings
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.vector_store import VectorStore
from app.tools.calculator import CalculatorTool
from app.tools.citation_tool import CitationTool
from app.tools.retrieval_tool import RetrievalTool


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(
        persist_path=settings.vector_db_absolute_path,
        collection_name=settings.collection_name,
    )


@lru_cache(maxsize=1)
def get_query_agent() -> QueryAgent:
    settings = get_settings()
    retriever = SemanticRetriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )
    retrieval_tool = RetrievalTool(retriever=retriever)

    return QueryAgent(
        settings=settings,
        retrieval_tool=retrieval_tool,
        calculator_tool=CalculatorTool(),
        citation_tool=CitationTool(),
    )
