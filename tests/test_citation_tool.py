from app.models.schemas import RetrievedChunk
from app.tools.citation_tool import CitationTool


def test_citation_tool_returns_quote_with_page() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="p033-txt000",
            page_number=33,
            source_type="text",
            section_title="Employment",
            table_id=None,
            content="The cybersecurity sector employs 7,351 people across Ireland in 2022.",
            score=0.93,
        )
    ]

    tool = CitationTool()
    citations = tool.run(
        query="What is the total number of jobs reported?",
        retrieved_chunks=chunks,
        limit=2,
    )

    assert len(citations) == 1
    assert citations[0].page_number == 33
    assert "7,351" in citations[0].quote
