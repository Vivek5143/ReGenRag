"""ReGenRAG FastAPI application entrypoint.

The application is assembled here and routers are mounted. Endpoints live in
their route modules; this file only wires them together.

Phase 3 wires in the baseline RAG query endpoint (retrieval + generation) on
top of the Phase 2 ingestion pipeline.
Phase 8 adds production API foundation: request ID middleware, global exception
handling, and standardized error responses.
Phase 8.2 adds observability: structured logging, metrics, health/readiness endpoints.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.api_router import api_router
from app.api.routes import health
from app.core.config import settings
from app.core.exceptions import (
    SessionNotFoundError,
    SessionNotActiveError,
    DocumentValidationError,
    DocumentProcessingError,
    NoProcessedDocumentError,
    RetrievalError,
    LLMConfigError,
    LLMError,
    ReGenRAGError,
)
from app.core.metrics import get_metrics
from app.core.middleware import (
    RequestIDMiddleware,
    create_error_response,
    document_validation_handler,
    document_processing_handler,
    general_exception_handler,
    http_exception_handler,
    llm_config_error_handler,
    llm_error_handler,
    no_processed_document_handler,
    regenrag_error_handler,
    retrieval_error_handler,
    request_validation_exception_handler,
    session_not_active_handler,
    session_not_found_handler,
)


def create_app() -> FastAPI:
    """Application factory used by uvicorn, tests, and ASGI servers."""
    app = FastAPI(
        title=settings.app_name,
        version="0.3.0",
        description="Self-healing RAG for evidence-grounded document intelligence.",
    )

    # Add request ID middleware
    app.add_middleware(RequestIDMiddleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handlers for standardized error responses
    app.add_exception_handler(SessionNotFoundError, session_not_found_handler)
    app.add_exception_handler(SessionNotActiveError, session_not_active_handler)
    app.add_exception_handler(DocumentValidationError, document_validation_handler)
    app.add_exception_handler(DocumentProcessingError, document_processing_handler)
    app.add_exception_handler(NoProcessedDocumentError, no_processed_document_handler)
    app.add_exception_handler(RetrievalError, retrieval_error_handler)
    app.add_exception_handler(LLMConfigError, llm_config_error_handler)
    app.add_exception_handler(LLMError, llm_error_handler)
    app.add_exception_handler(ReGenRAGError, regenrag_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # Metrics endpoint
    @app.get("/metrics")
    def metrics_endpoint():
        """Application metrics endpoint (JSON format)."""
        metrics = get_metrics()
        return JSONResponse(content=metrics_to_dict(metrics))

    app.include_router(health.router)
    app.include_router(api_router)

    return app


def metrics_to_dict(metrics) -> dict:
    """Convert metrics to a JSON-serializable dict."""
    import time

    def histogram_summary(h):
        values = h.get_values()
        if not values:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def counter_value(c):
        return c.get()

    return {
        "timestamp": time.time(),
        "http": {
            "requests_total": counter_value(metrics.http_requests_total),
            "request_duration_seconds": histogram_summary(metrics.http_request_duration_seconds),
            "requests_by_status": {str(k): counter_value(v) for k, v in metrics.http_requests_by_status.items()},
        },
        "rag": {
            "requests_total": counter_value(metrics.rag_requests_total),
            "request_duration_seconds": histogram_summary(metrics.rag_request_duration_seconds),
            "retrieval_latency_seconds": histogram_summary(metrics.rag_retrieval_latency_seconds),
            "grading_latency_seconds": histogram_summary(metrics.rag_grading_latency_seconds),
            "generation_latency_seconds": histogram_summary(metrics.rag_generation_latency_seconds),
            "grounding_latency_seconds": histogram_summary(metrics.rag_grounding_latency_seconds),
            "retry_count_total": counter_value(metrics.rag_retry_count_total),
            "healing_count_total": counter_value(metrics.rag_healing_count_total),
            "failure_by_category": {k: counter_value(v) for k, v in metrics.rag_failure_by_category.items()},
        },
    }


app = create_app()