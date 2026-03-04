from __future__ import annotations

from app.models.schemas import RetrievedChunk
from app.retrieval.retriever import SemanticRetriever


class RetrievalTool:
    name = "document_retrieval"
    description = "Semantic retrieval over PDF chunks with page-level metadata."

    def __init__(self, retriever: SemanticRetriever) -> None:
        self.retriever = retriever

    def run(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.retriever.search(query=query, top_k=top_k)
