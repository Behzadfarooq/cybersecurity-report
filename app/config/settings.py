from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    report_pdf_url: str = Field(
        default=(
            "https://cyberireland.ie/wp-content/uploads/2022/05/"
            "State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf"
        ),
        alias="REPORT_PDF_URL",
    )

    raw_pdf_path: str = Field(default="data/raw/cyber_ireland_2022.pdf", alias="RAW_PDF_PATH")
    chunks_path: str = Field(default="data/processed/chunks.jsonl", alias="CHUNKS_PATH")
    vector_db_path: str = Field(default="data/vectordb", alias="VECTOR_DB_PATH")
    collection_name: str = Field(default="cyber_ireland_2022", alias="COLLECTION_NAME")

    enable_ocr_fallback: bool = Field(default=True, alias="ENABLE_OCR_FALLBACK")
    ocr_dpi: int = Field(default=180, alias="OCR_DPI")

    top_k: int = Field(default=6, alias="TOP_K")
    max_agent_steps: int = Field(default=6, alias="MAX_AGENT_STEPS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def raw_pdf_absolute_path(self) -> Path:
        return (self.project_root / self.raw_pdf_path).resolve()

    @property
    def chunks_absolute_path(self) -> Path:
        return (self.project_root / self.chunks_path).resolve()

    @property
    def vector_db_absolute_path(self) -> Path:
        return (self.project_root / self.vector_db_path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
