from __future__ import annotations

import logging

from app.config.settings import Settings
from app.etl.chunker import build_chunks, save_chunks_jsonl
from app.etl.extractor import extract_pdf_content
from app.etl.pdf_loader import ensure_pdf_exists
from app.models.schemas import ChunkRecord

LOGGER = logging.getLogger(__name__)


class ETLPipeline:
    """Orchestrates ingestion: source PDF -> extracted pages -> chunked JSONL."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, force_download: bool = False) -> list[ChunkRecord]:
        pdf_path = ensure_pdf_exists(
            pdf_path=self.settings.raw_pdf_absolute_path,
            source_url=self.settings.report_pdf_url,
            force_download=force_download,
        )
        pages = extract_pdf_content(
            pdf_path,
            enable_ocr_fallback=self.settings.enable_ocr_fallback,
            ocr_dpi=self.settings.ocr_dpi,
        )
        chunks = build_chunks(pages)
        save_chunks_jsonl(chunks=chunks, output_path=self.settings.chunks_absolute_path)

        LOGGER.info(
            "ETL pipeline complete",
            extra={
                "pdf": str(pdf_path),
                "chunks_path": str(self.settings.chunks_absolute_path),
                "num_chunks": len(chunks),
            },
        )
        return chunks
