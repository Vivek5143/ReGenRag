"""ReGenRAG FastAPI application entrypoint.

The application is assembled here and routers are mounted. Endpoints live in
their route modules; this file only wires them together.

Phase 1: database + temporary document session infrastructure. The RAG pipeline
(ingestion, retrieval, generation) is still not wired in.
"""

from fastapi import FastAPI

from app.api.api_router import api_router
from app.api.routes import health
from app.core.config import settings


def create_app() -> FastAPI:
    """Application factory used by uvicorn, tests, and ASGI servers."""
    app = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        description="Self-healing RAG for evidence-grounded document intelligence.",
    )

    app.include_router(health.router)
    app.include_router(api_router)

    return app


app = create_app()