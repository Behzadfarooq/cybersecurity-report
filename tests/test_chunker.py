from app.etl.chunker import build_chunks
from app.models.schemas import ExtractedPage


def test_chunker_preserves_page_and_table_metadata() -> None:
    pages = [
        ExtractedPage(
            page_number=12,
            text="SECTION TITLE\n\nThe cybersecurity sector employs 7,351 people in Ireland.",
            tables=["| Region | Pure-Play % |\n| --- | --- |\n| South-West | 18% |"],
        )
    ]

    chunks = build_chunks(pages)

    assert chunks
    assert any(chunk.page_number == 12 for chunk in chunks)
    assert any(chunk.source_type == "table" for chunk in chunks)
    assert any("7,351" in chunk.content for chunk in chunks)
