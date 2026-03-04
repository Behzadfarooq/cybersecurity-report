from __future__ import annotations

from app.etl.extractor import _merge_ocr_text


def test_merge_ocr_text_appends_only_new_lines() -> None:
    base_text = "The cybersecurity sector employs 7,351 people.\nFigure 3.2 shows 33% dedicated firms."
    ocr_text = "Figure 3.2 shows 33% dedicated firms.\nSouth-West concentration is 41%.\nNational average is 33%."

    merged = _merge_ocr_text(base_text, ocr_text)

    assert "[ocr]" in merged
    assert "South-West concentration is 41%." in merged
    assert "National average is 33%." in merged
    assert merged.count("Figure 3.2 shows 33% dedicated firms.") == 1


def test_merge_ocr_text_returns_base_when_no_novel_ocr_lines() -> None:
    base_text = "Pure-play share is 33%.\nDiversified share is 67%."
    ocr_text = "  Pure-play   share is 33%.\nDiversified share is 67%."

    merged = _merge_ocr_text(base_text, ocr_text)

    assert merged == base_text
