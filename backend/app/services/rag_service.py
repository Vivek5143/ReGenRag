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

Phase 5:
- Answer grounding evaluation via LLM judge.

Phase 6:
- Self-healing retry loop: bounded recovery from retrieval and grounding failures.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from app.core.config import settings
from app.core.exceptions import NoProcessedDocumentError, RetrievalError
from app.core.logging import get_logger
from app.core.metrics import get_metrics, timer
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.generation import generator
from app.evaluation import grounding
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


class HealingAction(str, Enum):
    """Explicit names for self-healing recovery actions."""

    QUERY_REWRITE = "QUERY_REWRITE"
    RE_RETRIEVE = "RE_RETRIEVE"
    ANSWER_REGENERATE = "ANSWER_REGENERATE"
    CONTEXT_IMPROVEMENT = "CONTEXT_IMPROVEMENT"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"


@dataclass(frozen=True)
class HealingStep:
    """Record of one healing action taken during the self-healing loop."""

    attempt: int
    action: HealingAction
    failure_category: failure_classifier.RetrievalFailureCategory | None = None
    details: str = ""


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

    Phase 6 adds self-healing metadata:
    - ``healed``: True when recovery succeeded after an initial failure.
    - ``attempts``: Total attempts made (initial + retries).
    - ``healing_steps``: List of healing actions taken.
    - ``grounding_score``: Phase 5 grounding evaluation score [0, 1].
    - ``retry_exhausted``: True when all retries were used without success.
    """

    answer: str
    sources: list[RagSource] = field(default_factory=list)
    retrieval_grading: similarity_grader.RetrievalGrading | None = None
    llm_relevance: llm_grader.LlmRetrievalGrading | None = None
    rewritten_query: str | None = None
    failure_category: failure_classifier.RetrievalFailureCategory | None = None
    healed: bool = False
    attempts: int = 1
    healing_steps: list[HealingStep] = field(default_factory=list)
    grounding_score: float | None = None
    retry_exhausted: bool = False


_INSUFFICIENT_EVIDENCE_ANSWER = (
    "I couldn't find enough relevant information in the provided documents "
    "to answer this question reliably."
)


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


def _build_sources(chunks: list) -> list[RagSource]:
    """Convert retrieved chunks to RagSource list."""
    return [
        RagSource(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            score=chunk.score,
        )
        for chunk in chunks
    ]


def _record_healing_step(
    healing_steps: list[HealingStep],
    attempt: int,
    action: HealingAction,
    failure_category: failure_classifier.RetrievalFailureCategory | None = None,
    details: str = "",
) -> list[HealingStep]:
    """Return a new list with the healing step appended."""
    new_step = HealingStep(
        attempt=attempt,
        action=action,
        failure_category=failure_category,
        details=details,
    )
    return [*healing_steps, new_step]


def answer_question(
    db: DbSession,
    session_id: uuid.UUID,
    question: str,
) -> RagAnswer:
    """Answer ``question`` against the processed documents in ``session_id``.

    Implements a bounded self-healing loop (Phase 6):
    1. Retrieve evidence for the current query.
    2. Grade retrieval (similarity + LLM relevance).
    3. If insufficient: rewrite query, re-retrieve (bounded by MAX_RAG_RETRIES).
    4. If sufficient: build context, generate answer.
    5. Grade grounding (Phase 5).
    6. If grounding fails: improve query/context, re-retrieve, regenerate (bounded).
    7. Return answer or controlled failure when retries exhausted.

    Returns the answer and its sources plus self-healing metadata.

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

    original_question = question.strip()
    current_query = original_question
    max_retries = settings.max_rag_retries
    healing_steps: list[HealingStep] = []
    final_answer: str | None = None
    final_sources: list[RagSource] = []
    final_retrieval_grading: similarity_grader.RetrievalGrading | None = None
    final_llm_relevance: llm_grader.LlmRetrievalGrading | None = None
    final_rewritten_query: str | None = None
    final_failure_category: failure_classifier.RetrievalFailureCategory | None = None
    final_grounding_score: float | None = None
    retry_exhausted = False
    attempt_count = 0

    # Loop: attempt 0 = initial, attempt 1..max_retries = retries
    for attempt in range(max_retries + 1):
        attempt_count += 1
        logger.info(
            "self-healing attempt=%d max_retries=%d query=%r",
            attempt,
            max_retries,
            current_query,
        )

        # --- Retrieval ---
        try:
            retrieved_chunks = retriever.retrieve(
                db, session_id, current_query, top_k=settings.rag_top_k
            )
        except Exception as exc:
            logger.error(
                "retrieval failed session_id=%s attempt=%d error=%s",
                session_id,
                attempt,
                exc,
                exc_info=True,
            )
            raise RetrievalError(
                f"Retrieval failed for session {session_id}: {exc}"
            ) from exc

        # --- Retrieval Grading ---
        retrieval_grading = similarity_grader.grade_retrieval(retrieved_chunks)
        llm_relevance = llm_grader.grade_relevance(current_query, retrieved_chunks)
        current_sources = _build_sources(retrieved_chunks)

        # --- Check Retrieval Sufficiency ---
        # Hybrid sufficiency strategy:
        # - Similarity grading provides the primary signal
        # - LLM relevance can rescue similarity-insufficient retrieval when
        #   meaningful relevant evidence is identified
        # - NO_RESULTS is always a hard failure
        #
        # Decision order:
        #   1. NO_RESULTS  → hard failure, no rewrite
        #   2. Similarity sufficient → generate answer
        #   3. LLM relevance has meaningful evidence → generate answer (rescued)
        #   4. Otherwise → rewrite query, re-retrieve, retry

        # Classify the failure (used for NO_RESULTS check and healing metadata)
        if attempt == 0:
            failure_category = failure_classifier.classify_initial_failure(
                chunks=retrieved_chunks,
                retrieval_grading=retrieval_grading,
                llm_relevance=llm_relevance,
            )
        else:
            failure_category = failure_classifier.classify_retry_failure(
                chunks=retrieved_chunks,
                retrieval_grading=retrieval_grading,
                llm_relevance=llm_relevance,
            )

        # --- 1. NO_RESULTS is a hard failure: nothing was retrieved at all.
        # Rewriting against an empty result set cannot help, so skip rewriting
        # and the second retrieval entirely: controlled insufficient-evidence
        # response, no generation.
        if failure_category is failure_classifier.RetrievalFailureCategory.NO_RESULTS:
            # Persist metadata before breaking so the final response carries
            # the correct classification instead of falling back to the
            # default INSUFFICIENT_EVIDENCE at the bottom of the function.
            final_failure_category = failure_category
            final_retrieval_grading = retrieval_grading
            final_llm_relevance = llm_relevance
            final_sources = current_sources
            healing_steps = _record_healing_step(
                healing_steps,
                attempt=attempt_count,
                action=HealingAction.RETRY_EXHAUSTED,
                failure_category=failure_category,
                details="No results retrieved; query rewrite skipped",
            )
            retry_exhausted = True
            break

        # --- 2. Similarity sufficient → generate answer
        if retrieval_grading.sufficient:
            pass  # proceed to answer generation below

        # --- 3. LLM relevance can rescue similarity-insufficient retrieval
        # when meaningful relevant evidence is identified. When rescued, leave
        # the healing/retry branch immediately and proceed to answer generation.
        elif llm_relevance.has_relevant_evidence:
            failure_category = None  # retrieval rescued; treat as sufficient

        # Record the failure metadata (preserved in final response)
        # Note: final_failure_category is set AFTER rescue check so rescued
        # retrievals get failure_category=None in the final response.
        final_failure_category = failure_category
        final_retrieval_grading = retrieval_grading
        final_llm_relevance = llm_relevance
        final_sources = current_sources

        # --- 4. Neither similarity nor LLM relevance is sufficient → heal
        if failure_category is not None:
            # If we've exhausted retries, stop
            if attempt >= max_retries:
                healing_steps = _record_healing_step(
                    healing_steps,
                    attempt=attempt_count,
                    action=HealingAction.RETRY_EXHAUSTED,
                    failure_category=failure_category,
                    details="Max retries reached for retrieval failure",
                )
                retry_exhausted = True
                break

            # --- Healing: Rewrite Query ---
            healing_steps = _record_healing_step(
                healing_steps,
                attempt=attempt_count,
                action=HealingAction.QUERY_REWRITE,
                failure_category=failure_category,
                details=f"Retrieval insufficient: {retrieval_grading.reason}",
            )

            rewritten_query = query_rewriter.rewrite_query(
                current_query,
                chunks=retrieved_chunks,
                retrieval_grading=retrieval_grading,
                llm_relevance=llm_relevance,
            )

            if not isinstance(rewritten_query, str) or not rewritten_query.strip():
                healing_steps = _record_healing_step(
                    healing_steps,
                    attempt=attempt_count,
                    action=HealingAction.RETRY_EXHAUSTED,
                    failure_category=failure_category,
                    details="Query rewrite failed or produced empty output",
                )
                retry_exhausted = True
                break

            rewritten_query = rewritten_query.strip()
            final_rewritten_query = rewritten_query

            healing_steps = _record_healing_step(
                healing_steps,
                attempt=attempt_count,
                action=HealingAction.RE_RETRIEVE,
                failure_category=failure_category,
                details=f"Rewritten query: {rewritten_query}",
            )

            current_query = rewritten_query
            continue

        # --- Retrieval Sufficient: Generate Answer ---
        context = context_builder.build_context(retrieved_chunks)
        try:
            # Always use the original user question for generation,
            # not the potentially rewritten query.
            answer = generator.generate_answer(original_question, context)
        except Exception as exc:
            logger.error(
                "generation failed session_id=%s attempt=%d error=%s",
                session_id,
                attempt,
                exc,
                exc_info=True,
            )
            raise

        # --- Grounding Evaluation (Phase 5) ---
        grounding_result = grounding.grade_grounding(current_query, context, answer)
        final_grounding_score = grounding_result.score

        if grounding_result.score >= settings.grounding_threshold:
            # Success: grounded answer
            final_answer = answer
            final_sources = current_sources
            final_retrieval_grading = retrieval_grading
            final_llm_relevance = llm_relevance
            break

        # --- Grounding Failure: Healing ---
        final_failure_category = None  # Retrieval was sufficient; failure is grounding
        final_retrieval_grading = retrieval_grading
        final_llm_relevance = llm_relevance
        final_sources = current_sources
        final_answer = answer  # Keep the last answer as fallback

        # Record grounding failure
        healing_steps = _record_healing_step(
            healing_steps,
            attempt=attempt_count,
            action=HealingAction.ANSWER_REGENERATE,
            details=f"Grounding score {grounding_result.score:.2f} below threshold {settings.grounding_threshold}",
        )

        if attempt >= max_retries:
            healing_steps = _record_healing_step(
                healing_steps,
                attempt=attempt_count,
                action=HealingAction.RETRY_EXHAUSTED,
                details="Max retries reached for grounding failure",
            )
            retry_exhausted = True
            break

        # Healing: Improve context by rewriting query and re-retrieving
        healing_steps = _record_healing_step(
            healing_steps,
            attempt=attempt_count,
            action=HealingAction.CONTEXT_IMPROVEMENT,
            details="Grounding failure; rewriting query to improve retrieval",
        )

        rewritten_query = query_rewriter.rewrite_query(
            current_query,
            chunks=retrieved_chunks,
            retrieval_grading=retrieval_grading,
            llm_relevance=llm_relevance,
        )

        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            healing_steps = _record_healing_step(
                healing_steps,
                attempt=attempt_count,
                action=HealingAction.RETRY_EXHAUSTED,
                details="Query rewrite failed during grounding recovery",
            )
            retry_exhausted = True
            break

        rewritten_query = rewritten_query.strip()
        final_rewritten_query = rewritten_query

        healing_steps = _record_healing_step(
            healing_steps,
            attempt=attempt_count,
            action=HealingAction.RE_RETRIEVE,
            details=f"Rewritten query for grounding recovery: {rewritten_query}",
        )

        current_query = rewritten_query
        continue

    # --- Final Result Construction ---
    if final_answer is None:
        final_answer = _INSUFFICIENT_EVIDENCE_ANSWER
        final_failure_category = final_failure_category or failure_classifier.RetrievalFailureCategory.INSUFFICIENT_EVIDENCE

    healed = len(healing_steps) > 0 and not retry_exhausted

    return RagAnswer(
        answer=final_answer,
        sources=final_sources,
        retrieval_grading=final_retrieval_grading,
        llm_relevance=final_llm_relevance,
        rewritten_query=final_rewritten_query,
        failure_category=final_failure_category,
        healed=healed,
        attempts=attempt_count,
        healing_steps=healing_steps,
        grounding_score=final_grounding_score,
        retry_exhausted=retry_exhausted,
    )