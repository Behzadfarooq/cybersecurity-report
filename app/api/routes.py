from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.agents.query_agent import QueryAgent
from app.api.dependencies import get_query_agent, get_vector_store
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check(vector_store: VectorStore = Depends(get_vector_store)) -> dict[str, str | int]:
    return {"status": "ok", "indexed_chunks": vector_store.count()}


@router.post("/query", response_model=QueryResponse)
def query_report(
    request: QueryRequest,
    agent: QueryAgent = Depends(get_query_agent),
    vector_store: VectorStore = Depends(get_vector_store),
) -> QueryResponse:
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector index is empty. Run the ingestion script first: "
                "python -m scripts.ingest_report"
            ),
        )

    try:
        result = agent.answer(request.query)
    except Exception as exc:
        LOGGER.exception("Query handling failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pages = sorted({citation.page_number for citation in result.citations})
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        page=pages,
        reasoning_steps=result.reasoning_steps,
    )
