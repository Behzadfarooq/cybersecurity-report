from fastapi.testclient import TestClient

from app.api.dependencies import get_query_agent, get_vector_store
from app.main import create_app
from app.models.schemas import AgentResult, Citation, ReasoningStep


class FakeVectorStore:
    def count(self) -> int:
        return 10


class FakeAgent:
    def answer(self, query: str) -> AgentResult:
        return AgentResult(
            answer=f"Echo answer for: {query}",
            citations=[
                Citation(
                    chunk_id="p033-txt000",
                    page_number=33,
                    quote="The cybersecurity sector employs 7,351 people.",
                    source_type="text",
                )
            ],
            reasoning_steps=[
                ReasoningStep(
                    step_number=1,
                    action="retrieve",
                    detail="Fake retrieval",
                    tool_input={"query": query},
                    tool_output={"count": 1},
                )
            ],
        )


def test_query_endpoint_response_shape() -> None:
    app = create_app()
    app.dependency_overrides[get_vector_store] = FakeVectorStore
    app.dependency_overrides[get_query_agent] = FakeAgent

    client = TestClient(app)
    response = client.post("/query", json={"query": "test question"})

    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "citations" in payload
    assert "page" in payload
    assert "reasoning_steps" in payload
    assert payload["page"] == [33]
