"""Baseline RAG orchestration.

Wires the Phase 3 pipeline end-to-end for a single query:

    question -> embed -> pgvector retrieval -> context -> LLM -> answer + sources

Keeps the pieces (retriever, context builder, generator) decoupled; this service
only validates the session and composes them. All failure modes surface as
domain exceptions that the API route maps to HTTP statuses.

Phase 4 (partial):
- Evidence gating decides whether generation is allowed.
- When evidence is insufficient, we rewrite the query for a later
  re-retrieval phase (Phase 4 Step 3).
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
from app.retrieval import (
    context_builder,
    failure_classifier,
    llm_grader,
    query_rewriter,
    retriever,
    similarity_grader,
)
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
    """A generated answer plus the chunks it was grounded in.

    ``retrieval_grading`` reports whether the retrieved evidence met the
    similarity threshold; ``llm_relevance`` adds per-chunk semantic relevance
    judgements.

    For insufficient evidence, Phase 4 produces a ``rewritten_query`` for a
    later re-retrieval phase (Phase 4 Step 3).

    Note: This service only implements Phase 4 up to query rewriting + a
    single re-retrieval attempt; the overall self-healing pipeline continues
    in later phases.
    """

    answer: str
    sources: list[RagSource] = field(default_factory=list)
    retrieval_grading: similarity_grader.RetrievalGrading | None = None
    llm_relevance: llm_grader.LlmRetrievalGrading | None = None
    rewritten_query: str | None = None
    failure_category: failure_classifier.RetrievalFailureCategory | None = None


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


_INSUFFICIENT_EVIDENCE_ANSWER = (
    "I couldn't find enough relevant information in the provided documents "
    "to answer this question reliably."
)


def answer_question(db: DbSession, session_id: uuid.UUID, question: str) -> RagAnswer:
    """Answer ``question`` against the processed documents in ``session_id``.

    Returns the answer and its sources.

    Raises ``SessionNotFoundError`` for an unknown session,
    ``NoProcessedDocumentError`` when nothing is searchable,
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
        initial_chunks = retriever.retrieve(
            db, session_id, question, top_k=settings.rag_top_k
        )
    except Exception as exc:
        logger.error(
            "retrieval failed session_id=%s error=%s", session_id, exc, exc_info=True
        )
        raise RetrievalError(
            f"Retrieval failed for session {session_id}: {exc}"
        ) from exc

    initial_retrieval_grading = similarity_grader.grade_retrieval(initial_chunks)
    initial_llm_relevance = llm_grader.grade_relevance(
        question, initial_chunks
    )

    initial_sources = [
        RagSource(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            score=chunk.score,
        )
        for chunk in initial_chunks
    ]

    # Successful initial retrieval => baseline behavior: context + generate.
    if initial_retrieval_grading.sufficient:
        context = context_builder.build_context(initial_chunks)
        answer = generator.generate_answer(question, context)
        return RagAnswer(
            answer=answer,
            sources=initial_sources,
            retrieval_grading=initial_retrieval_grading,
            llm_relevance=initial_llm_relevance,
            rewritten_query=None,
            failure_category=None,
        )

    # Phase 4 Step 2: rewrite query for a single re-retrieval attempt.
    initial_failure_category = failure_classifier.classify_initial_failure(
        chunks=initial_chunks,
        retrieval_grading=initial_retrieval_grading,
        llm_relevance=initial_llm_relevance,
    )

    # NO_RESULTS means nothing was retrieved at all. Rewriting a query against
    # an empty result set cannot help, so skip rewriting and the second
    # retrieval entirely: controlled insufficient-evidence response, no
    # generation. Other insufficient categories proceed to rewrite + re-retrieve.
    if initial_failure_category is failure_classifier.RetrievalFailureCategory.NO_RESULTS:
        return RagAnswer(
            answer=_INSUFFICIENT_EVIDENCE_ANSWER,
            sources=initial_sources,
            retrieval_grading=initial_retrieval_grading,
            llm_relevance=initial_llm_relevance,
            rewritten_query=None,
            failure_category=initial_failure_category,
        )

    rewritten_query = query_rewriter.rewrite_query(
        question,
        chunks=initial_chunks,
        retrieval_grading=initial_retrieval_grading,
        llm_relevance=initial_llm_relevance,
    )

    # If rewrite failed or produced empty/invalid output => stop.
    if not isinstance(rewritten_query, str) or not rewritten_query.strip():
        return RagAnswer(
            answer=_INSUFFICIENT_EVIDENCE_ANSWER,
            sources=initial_sources,
            retrieval_grading=initial_retrieval_grading,
            llm_relevance=initial_llm_relevance,
            rewritten_query=None,
            failure_category=initial_failure_category,
        )

    rewritten_query = rewritten_query.strip()

    # Phase 4 Step 3: re-retrieve once using the rewritten query.
    try:
        retry_chunks = retriever.retrieve(
            db, session_id, rewritten_query, top_k=settings.rag_top_k
        )
    except Exception as exc:
        # Re-retrieval failure is still a retrieval failure.
        logger.error(
            "retrieval (rewritten_query) failed session_id=%s error=%s",
            session_id,
            exc,
            exc_info=True,
        )
        raise RetrievalError(
            f"Retrieval failed for session {session_id}: {exc}"
        ) from exc

    retry_retrieval_grading = similarity_grader.grade_retrieval(retry_chunks)
    retry_llm_relevance = llm_grader.grade_relevance(question, retry_chunks)

    retry_sources = [
        RagSource(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            score=chunk.score,
        )
        for chunk in retry_chunks
    ]

    if retry_retrieval_grading.sufficient:
        context = context_builder.build_context(retry_chunks)
        answer = generator.generate_answer(question, context)
        return RagAnswer(
            answer=answer,
            sources=retry_sources,
            retrieval_grading=retry_retrieval_grading,
            llm_relevance=retry_llm_relevance,
            rewritten_query=rewritten_query,
            failure_category=None,
        )

    # Controlled insufficient-evidence response after the second attempt.
    retry_failure_category = failure_classifier.classify_retry_failure(
        chunks=retry_chunks,
        retrieval_grading=retry_retrieval_grading,
        llm_relevance=retry_llm_relevance,
    )

    return RagAnswer(
        answer=_INSUFFICIENT_EVIDENCE_ANSWER,
        sources=retry_sources,
        retrieval_grading=retry_retrieval_grading,
        llm_relevance=retry_llm_relevance,
        rewritten_query=rewritten_query,
        failure_category=retry_failure_category,
    )
