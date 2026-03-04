from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import fitz
import pdfplumber

from app.models.schemas import ExtractedPage

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - optional runtime dependency
    RapidOCR = None

LOGGER = logging.getLogger(__name__)
_OCR_LINE_CHAR_BUDGET = 3500
_OCR_MAX_LINES = 80
_RAPIDOCR_ENGINE: Any | None = None
_OCR_UNAVAILABLE_LOGGED = False
_OCR_RUNTIME_FAILURE_LOGGED = False


def _normalize_table_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    compact = re.sub(r"\s+", " ", str(cell)).strip()
    return compact


def _table_to_markdown(table_rows: list[list[str | None]]) -> str:
    rows = [
        [_normalize_table_cell(cell) for cell in row]
        for row in table_rows
        if row and any(cell is not None and str(cell).strip() for cell in row)
    ]

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized_rows: list[list[str]] = []
    for row in rows:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        normalized_rows.append(row)

    header = normalized_rows[0]
    body = normalized_rows[1:]

    markdown_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    markdown_lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(markdown_lines)


def _normalize_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _get_ocr_engine() -> Any | None:
    global _RAPIDOCR_ENGINE
    global _OCR_UNAVAILABLE_LOGGED

    if RapidOCR is None:
        if not _OCR_UNAVAILABLE_LOGGED:
            LOGGER.warning(
                "rapidocr-onnxruntime is not installed; OCR fallback is disabled.",
            )
            _OCR_UNAVAILABLE_LOGGED = True
        return None

    if _RAPIDOCR_ENGINE is None:
        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


def _extract_page_ocr_text(page: fitz.Page, dpi: int) -> str:
    global _OCR_RUNTIME_FAILURE_LOGGED

    engine = _get_ocr_engine()
    if engine is None:
        return ""

    try:
        zoom = max(float(dpi), 72.0) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        raw_result = engine(pix.tobytes("png"))
    except Exception as exc:  # pragma: no cover - runtime/environment specific
        if not _OCR_RUNTIME_FAILURE_LOGGED:
            LOGGER.warning(
                "OCR fallback failed during page processing; continuing without OCR.",
                extra={"error": str(exc)},
            )
            _OCR_RUNTIME_FAILURE_LOGGED = True
        return ""

    if isinstance(raw_result, tuple):
        ocr_lines = raw_result[0]
    else:
        ocr_lines = raw_result

    if not ocr_lines:
        return ""

    cleaned_lines: list[str] = []
    for line in ocr_lines:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        if not isinstance(line[1], str):
            continue
        compact = re.sub(r"\s+", " ", line[1]).strip()
        if compact:
            cleaned_lines.append(compact)

    return "\n".join(cleaned_lines)


def _merge_ocr_text(base_text: str, ocr_text: str) -> str:
    if not ocr_text:
        return base_text

    existing_lines = {
        _normalize_line(line)
        for line in base_text.splitlines()
        if line.strip()
    }

    selected_lines: list[str] = []
    char_total = 0
    for raw_line in ocr_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < 3:
            continue
        if re.fullmatch(r"[\W_]+", line):
            continue

        normalized = _normalize_line(line)
        if not normalized or normalized in existing_lines:
            continue

        if len(selected_lines) >= _OCR_MAX_LINES:
            break
        if char_total + len(line) > _OCR_LINE_CHAR_BUDGET:
            break

        selected_lines.append(line)
        existing_lines.add(normalized)
        char_total += len(line)

    if not selected_lines:
        return base_text

    ocr_block = "\n".join(selected_lines)
    if not base_text:
        return f"[ocr]\n{ocr_block}"
    return f"{base_text}\n\n[ocr]\n{ocr_block}"


def extract_pdf_content(
    pdf_path: Path,
    enable_ocr_fallback: bool = True,
    ocr_dpi: int = 180,
) -> list[ExtractedPage]:
    """Extract page text and tables from a PDF."""

    pages: list[ExtractedPage] = []
    ocr_augmented_pages = 0

    with fitz.open(str(pdf_path)) as fitz_doc, pdfplumber.open(str(pdf_path)) as pdf_doc:
        for page_idx, page in enumerate(pdf_doc.pages, start=1):
            fitz_page = fitz_doc[page_idx - 1]
            text = (page.extract_text(layout=True) or "").strip()

            # Fallback for pages where pdfplumber text extraction is sparse.
            if not text:
                text = fitz_page.get_text("text").strip()

            if enable_ocr_fallback:
                ocr_text = _extract_page_ocr_text(page=fitz_page, dpi=ocr_dpi)
                merged_text = _merge_ocr_text(base_text=text, ocr_text=ocr_text)
                if merged_text != text:
                    ocr_augmented_pages += 1
                text = merged_text

            tables: list[str] = []
            raw_tables = page.extract_tables() or []
            for raw_table in raw_tables:
                table_markdown = _table_to_markdown(raw_table)
                if table_markdown:
                    tables.append(table_markdown)

            pages.append(
                ExtractedPage(
                    page_number=page_idx,
                    text=text,
                    tables=tables,
                )
            )

    LOGGER.info(
        "PDF extraction complete",
        extra={
            "pages": len(pages),
            "pdf_path": str(pdf_path),
            "ocr_fallback_enabled": enable_ocr_fallback,
            "ocr_augmented_pages": ocr_augmented_pages,
        },
    )
    return pages
