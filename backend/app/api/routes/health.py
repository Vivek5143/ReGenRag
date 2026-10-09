"""Health check endpoints: liveness and readiness.

/health  - liveness only (does not check database or external services)
/ready   - readiness (checks database connectivity)
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.schemas import HealthResponse
from app.core.config import settings
from app.core.logging import get_logger
from app.db.database import get_engine

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


def _database_connected() -> bool:
    """Return whether the database is reachable (without exposing credentials)."""
    try:
        engine = get_engine()
    except RuntimeError:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("database health check failed")
        return False


@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness endpoint. Does NOT check database or external services.

    Returns 200 if the process is alive.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "service": settings.app_name,
        },
    )


@router.get("/ready", response_model=HealthResponse)
def ready():
    """Readiness endpoint. Checks database connectivity.

    Returns 200 if ready to serve requests, 503 if database unavailable.
    """
    if _database_connected():
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "service": settings.app_name,
                "database": "connected",
            },
        )
    return JSONResponse(
        status_code=503,
        content={
            "status": "degraded",
            "service": settings.app_name,
            "database": "disconnected",
        },
    )