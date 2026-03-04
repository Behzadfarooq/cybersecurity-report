from __future__ import annotations

import logging
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)


def download_pdf(url: str, destination: Path, timeout_seconds: float = 120.0) -> Path:
    """Download a PDF file from a URL to a local destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with destination.open("wb") as file_handle:
            for chunk in response.iter_bytes():
                file_handle.write(chunk)

    LOGGER.info("PDF downloaded", extra={"url": url, "destination": str(destination)})
    return destination


def ensure_pdf_exists(pdf_path: Path, source_url: str, force_download: bool = False) -> Path:
    """Ensure the source PDF exists locally; download if needed."""

    if force_download or not pdf_path.exists():
        LOGGER.info("Downloading source PDF", extra={"force_download": force_download})
        return download_pdf(url=source_url, destination=pdf_path)

    LOGGER.info("Using existing local PDF", extra={"path": str(pdf_path)})
    return pdf_path
