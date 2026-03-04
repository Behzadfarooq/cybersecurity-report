from app.agents.query_agent import QueryAgent
from app.config.settings import Settings
from app.models.schemas import RetrievedChunk
from app.tools.calculator import CalculatorTool
from app.tools.citation_tool import CitationTool


class FakeRetrievalTool:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def run(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._chunks[:top_k]


def test_agent_executes_cagr_reasoning_with_calculator_tool() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="p033-txt000",
            page_number=33,
            source_type="text",
            section_title="Employment",
            table_id=None,
            content="The cybersecurity sector employs 7,351 people in 2022.",
            score=0.95,
        ),
        RetrievedChunk(
            chunk_id="p010-txt001",
            page_number=10,
            source_type="text",
            section_title="Forecast",
            table_id=None,
            content="By 2030, the cluster could support employment of over 17,000 professionals.",
            score=0.89,
        ),
    ]

    settings = Settings(OPENAI_API_KEY=None, TOP_K=6, MAX_AGENT_STEPS=6)

    agent = QueryAgent(
        settings=settings,
        retrieval_tool=FakeRetrievalTool(chunks),
        calculator_tool=CalculatorTool(),
        citation_tool=CitationTool(),
    )

    result = agent.answer(
        "Based on our 2022 baseline and the stated 2030 job target, what is the required CAGR?"
    )

    calculation_steps = [step for step in result.reasoning_steps if step.action == "calculate"]
    assert calculation_steps

    calc_value = calculation_steps[0].tool_output["value"]
    assert abs(calc_value - 0.1104852958) < 1e-6
    assert "11.05%" in result.answer
