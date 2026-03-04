from __future__ import annotations

import json
import re
from pathlib import Path

from app.models.schemas import ChunkRecord, ExtractedPage


def clean_page_text(text: str) -> str:
    """Normalize noisy PDF text while preserving sentence boundaries."""

    normalized = text.replace("\u00a0", " ")
    normalized = re.sub(r"-\n(?=[a-z])", "", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def detect_section_title(page_text: str) -> str | None:
    """Heuristic section detector based on short title-like first lines."""

    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    for line in lines[:5]:
        alpha_ratio = sum(char.isalpha() for char in line) / max(len(line), 1)
        if len(line) <= 80 and alpha_ratio > 0.5 and line[:1].isupper():
            if line.isupper() or line.endswith(":"):
                return line.title()
    return None


def _split_text_into_chunks(text: str, max_chars: int = 1100, overlap_chars: int = 180) -> list[str]:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        projected = current_len + len(paragraph) + (2 if current_parts else 0)
        if projected > max_chars and current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append(chunk_text)

            if overlap_chars > 0:
                overlap = chunk_text[-overlap_chars:].strip()
                current_parts = [overlap, paragraph] if overlap else [paragraph]
                current_len = sum(len(part) for part in current_parts) + (2 * (len(current_parts) - 1))
            else:
                current_parts = [paragraph]
                current_len = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_len = projected

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def _split_table_if_needed(table_markdown: str, max_chars: int = 1600) -> list[str]:
    if len(table_markdown) <= max_chars:
        return [table_markdown]

    lines = table_markdown.splitlines()
    if len(lines) <= 4:
        return [table_markdown]

    header = lines[:2]
    body = lines[2:]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for row in body:
        if current_len + len(row) + 1 > max_chars and current:
            chunk = "\n".join(header + current)
            chunks.append(chunk)
            current = [row]
            current_len = len(row)
        else:
            current.append(row)
            current_len += len(row) + 1

    if current:
        chunks.append("\n".join(header + current))

    return chunks


def build_chunks(pages: list[ExtractedPage]) -> list[ChunkRecord]:
    """Create chunk records from extracted pages with rich metadata."""

    chunks: list[ChunkRecord] = []

    for page in pages:
        cleaned_text = clean_page_text(page.text)
        section_title = detect_section_title(cleaned_text) if cleaned_text else None

        text_chunks = _split_text_into_chunks(cleaned_text) if cleaned_text else []
        for idx, chunk_text in enumerate(text_chunks):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"p{page.page_number:03d}-txt{idx:03d}",
                    page_number=page.page_number,
                    source_type="text",
                    section_title=section_title,
                    table_id=None,
                    content=chunk_text,
                    token_count=len(chunk_text.split()),
                )
            )

        for table_idx, table in enumerate(page.tables):
            table_parts = _split_table_if_needed(table)
            for part_idx, table_part in enumerate(table_parts):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"p{page.page_number:03d}-tbl{table_idx:03d}-part{part_idx:03d}",
                        page_number=page.page_number,
                        source_type="table",
                        section_title=section_title,
                        table_id=f"table_{page.page_number}_{table_idx}",
                        content=table_part,
                        token_count=len(table_part.split()),
                    )
                )

    return chunks


def save_chunks_jsonl(chunks: list[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(), ensure_ascii=True) + "\n")


def load_chunks_jsonl(path: Path) -> list[ChunkRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {path}")

    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(ChunkRecord.model_validate_json(line))
    return chunks
