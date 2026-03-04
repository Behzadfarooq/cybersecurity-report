from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    page_number: int
    text: str
    tables: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    page_number: int
    source_type: Literal["text", "table"]
    section_title: str | None = None
    table_id: str | None = None
    content: str
    token_count: int


class RetrievedChunk(BaseModel):
    chunk_id: str
    page_number: int
    source_type: Literal["text", "table"]
    section_title: str | None = None
    table_id: str | None = None
    content: str
    score: float


class Citation(BaseModel):
    chunk_id: str
    page_number: int
    quote: str
    source_type: Literal["text", "table"]


class ReasoningStep(BaseModel):
    step_number: int
    trace_id: str | None = None
    action: str
    detail: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: dict[str, Any] = Field(default_factory=dict)
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    page: list[int]
    reasoning_steps: list[ReasoningStep]


class AgentResult(BaseModel):
    answer: str
    citations: list[Citation]
    reasoning_steps: list[ReasoningStep]


class PlannerDecision(BaseModel):
    action: Literal["retrieve", "calculate", "cite", "finish"]
    action_input: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
