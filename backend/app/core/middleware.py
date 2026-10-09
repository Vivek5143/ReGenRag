"""Request ID middleware and global exception handlers."""

import time
import uuid
from typing import Callable

from fastapi import Request, Response, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.exceptions import (
    ReGenRAGError,
    SessionNotFoundError,
    SessionNotActiveError,
    DocumentValidationError,
    DocumentProcessingError,
    NoProcessedDocumentError,
    RetrievalError,
    LLMConfigError,
    LLMError,
)
from app.core.logging import get_logger
from app.core.metrics import get_metrics, timer

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and attach request ID to each request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Record metrics
        metrics = get_metrics()
        metrics.http_requests_total.increment()
        metrics.http_request_duration_seconds.observe(duration_ms / 1000.0)
        metrics._get_status_counter(response.status_code).increment()

        # Log request completion with timing
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host if request.client else None,
            },
        )

        return response


def create_error_response(
    request: Request,
    status_code: int,
    error_type: str,
    message: str,
    detail: str | None = None,
) -> JSONResponse:
    """Create standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
                "detail": detail,
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": time.time(),
            }
        },
    )


# Exception handlers
async def session_not_found_handler(request: Request, exc: SessionNotFoundError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=404,
        error_type="SessionNotFoundError",
        message=str(exc),
        detail=str(exc),
    )


async def session_not_active_handler(request: Request, exc: SessionNotActiveError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=409,
        error_type="SessionNotActiveError",
        message=str(exc),
        detail=str(exc),
    )


async def document_validation_handler(request: Request, exc: DocumentValidationError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=400,
        error_type="DocumentValidationError",
        message=str(exc),
        detail=str(exc),
    )


async def document_processing_handler(request: Request, exc: DocumentProcessingError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=500,
        error_type="DocumentProcessingError",
        message=str(exc),
        detail=str(exc),
    )


async def no_processed_document_handler(request: Request, exc: NoProcessedDocumentError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=404,
        error_type="NoProcessedDocumentError",
        message=str(exc),
        detail=str(exc),
    )


async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=500,
        error_type="RetrievalError",
        message=str(exc),
        detail=str(exc),
    )


async def llm_config_error_handler(request: Request, exc: LLMConfigError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=503,
        error_type="LLMConfigError",
        message=str(exc),
        detail=str(exc),
    )


async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=502,
        error_type="LLMError",
        message=str(exc),
        detail=str(exc),
    )


async def regenrag_error_handler(request: Request, exc: ReGenRAGError) -> JSONResponse:
    return create_error_response(
        request,
        status_code=500,
        error_type="ReGenRAGError",
        message=str(exc),
        detail=str(exc),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return create_error_response(
        request,
        status_code=exc.status_code,
        error_type="HTTPException",
        message=exc.detail,
        detail=getattr(exc, "detail", None),
    )


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # RequestValidationError doesn't have status_code, but we can use 422 (Unprocessable Entity)
    # Extract the error details for better debugging
    error_details = []
    for error in exc.errors():
        error_details.append(f"{'.'.join(str(x) for x in error['loc'])}: {error['msg']}")

    message = "Validation error: " + "; ".join(error_details)

    return create_error_response(
        request,
        status_code=422,
        error_type="RequestValidationError",
        message=message,
        detail=str(exc),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log unexpected exceptions for debugging
    logger.error(
        "unhandled exception",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "exception": str(exc),
            "exception_type": type(exc).__name__,
        },
        exc_info=True,
    )
    return create_error_response(
        request,
        status_code=500,
        error_type="InternalServerError",
        message="An internal server error occurred",
        detail=None if settings.app_env == "production" else str(exc),
    )