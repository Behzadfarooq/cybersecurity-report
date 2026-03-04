from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config.logging import setup_logging
from app.config.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    application = FastAPI(
        title="Cyber Ireland Report Agent API",
        version="1.0.0",
        description="Headless backend for grounded QA over the Cyber Ireland 2022 report.",
    )
    application.include_router(api_router)
    return application


app = create_app()
