"""Baseline RAG orchestration.

Wires the Phase 3 pipeline end-to-end for a single query:

    question -> embed -> pgvector retrieval -> context -> LLM -> answer + sources

Keeps the pieces (retriever, context builder, generator) decoupled; this service
only validates the session and composes them. All failure modes surface as
domain exceptions that the API route maps to HTTP statuses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from app.core.config import settings
from app.core.exceptions import NoProcessedDocumentError, RetrievalError
from app.core.logging import get_logger
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.generation import generator
from app.retrieval import context_builder, retriever
from app.services import session_service

logger = get_logger(__name__)


@dataclass(frozen=True)
class RagSource:
    """Source metadata for one retrieved chunk backing an answer."""

    document_id: uuid.UUID
    chunk_id: uuid.UUID
    chunk_index: int
    page_number: int | None
    score: float | None


@dataclass(frozen=True)
class RagAnswer:
    """A generated answer plus the chunks it was grounded in."""

    answer: str
    sources: list[RagSource] = field(default_factory=list)


def _ensure_processed_document(db: DbSession, session_id: uuid.UUID) -> None:
    """Raise if the session has no PROCESSED document to query against."""
    count = db.scalar(
        select(func.count())
        .select_from(Document)
        .where(
            Document.session_id == session_id,
            Document.status == DocumentStatus.PROCESSED,
        )
    ) or 0
    if count == 0:
        raise NoProcessedDocumentError(
            f"Session {session_id} has no processed document to query"
        )


def answer_question(db: DbSession, session_id: uuid.UUID, question: str) -> RagAnswer:
    """Answer ``question`` against the processed documents in ``session_id``.

    Returns the answer and its sources. Raises ``SessionNotFoundError`` for an
    unknown session, ``NoProcessedDocumentError`` when nothing is searchable,
    ``RetrievalError`` on a retrieval/database failure, and ``LLMError`` /
    ``LLMConfigError`` on generation failure.
    """
    session = session_service.get_session(db, session_id)

    # An empty/blank question is rejected by the request schema, but guard here
    # so the service is safe to call directly too.
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    _ensure_processed_document(db, session_id)
    # A query is an interaction: extend the session's rolling expiry.
    session_service.touch_session(db, session)

    try:
        chunks = retriever.retrieve(
            db, session_id, question, top_k=settings.rag_top_k
        )
    except Exception as exc:
        logger.error(
            "retrieval failed session_id=%s error=%s", session_id, exc, exc_info=True
        )
        raise RetrievalError(
            f"Retrieval failed for session {session_id}: {exc}"
        ) from exc

    context = context_builder.build_context(chunks)
    answer = generator.generate_answer(question, context)

    sources = [
        RagSource(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            score=chunk.score,
        )
        for chunk in chunks
    ]
    return RagAnswer(answer=answer, sources=sources)
