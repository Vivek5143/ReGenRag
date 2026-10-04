"""Unit tests for retrieval failure classification (Phase 4, step 4).

Tests the actual categories and precedence implemented in
``app.retrieval.failure_classifier``.
"""

from __future__ import annotations

import uuid

from app.retrieval.failure_classifier import (
    RetrievalFailureCategory,
    classify_initial_failure,
    classify_retry_failure,
)
from app.retrieval.llm_grader import LlmRetrievalGrading
from app.retrieval.similarity_grader import RetrievalGrading
from app.retrieval.vector_store import RetrievedChunk


def _chunk(score: float = 0.3) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content="evidence",
        score=score,
    )


def _grading(sufficient: bool, threshold: float = 0.65) -> RetrievalGrading:
    return RetrievalGrading(
        sufficient=sufficient,
        threshold=threshold,
        relevant_chunks=[],
        low_relevance_chunks=[],
        best_score=None,
        average_score=None,
        reason="test",
    )


def _llm_relevance(has_relevant_evidence: bool) -> LlmRetrievalGrading:
    return LlmRetrievalGrading(
        judgements=[],
        relevant_count=1 if has_relevant_evidence else 0,
        failed_count=0,
        has_relevant_evidence=has_relevant_evidence,
        reason="ok" if has_relevant_evidence else "no relevant evidence",
    )


# ---------------------------------------------------------------------------
# classify_initial_failure
# ---------------------------------------------------------------------------


def test_initial_no_results_when_no_chunks():
    result = classify_initial_failure(
        chunks=[],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
    )
    assert result == RetrievalFailureCategory.NO_RESULTS


def test_initial_returns_none_when_sufficient():
    result = classify_initial_failure(
        chunks=[_chunk(0.9)],
        retrieval_grading=_grading(sufficient=True),
        llm_relevance=_llm_relevance(has_relevant_evidence=True),
    )
    assert result is None


def test_initial_low_similarity_when_llm_says_relevant():
    result = classify_initial_failure(
        chunks=[_chunk(0.3)],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=True),
    )
    assert result == RetrievalFailureCategory.LOW_SIMILARITY


def test_initial_low_llm_relevance_when_llm_says_not_relevant():
    result = classify_initial_failure(
        chunks=[_chunk(0.3)],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
    )
    assert result == RetrievalFailureCategory.LOW_LLM_RELEVANCE


def test_initial_precedence_no_results_beats_low_similarity():
    # NO_RESULTS is checked first: even with an LLM verdict, empty chunks win.
    result = classify_initial_failure(
        chunks=[],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=True),
    )
    assert result == RetrievalFailureCategory.NO_RESULTS


def test_initial_precedence_sufficient_beats_all_failure_labels():
    # Sufficient evidence means no failure, regardless of LLM verdict.
    result = classify_initial_failure(
        chunks=[_chunk(0.3)],
        retrieval_grading=_grading(sufficient=True),
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
    )
    assert result is None


def test_initial_precedence_low_similarity_beats_low_llm_relevance():
    # When similarity failed but the LLM found relevant evidence, the label is
    # LOW_SIMILARITY, not LOW_LLM_RELEVANCE.
    result = classify_initial_failure(
        chunks=[_chunk(0.3)],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=True),
    )
    assert result == RetrievalFailureCategory.LOW_SIMILARITY


# ---------------------------------------------------------------------------
# classify_retry_failure
# ---------------------------------------------------------------------------


def test_retry_no_results_when_no_chunks():
    result = classify_retry_failure(
        chunks=[],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
    )
    assert result == RetrievalFailureCategory.NO_RESULTS


def test_retry_returns_none_when_sufficient():
    result = classify_retry_failure(
        chunks=[_chunk(0.9)],
        retrieval_grading=_grading(sufficient=True),
        llm_relevance=_llm_relevance(has_relevant_evidence=True),
    )
    assert result is None


def test_retry_insufficient_evidence_when_similarity_failed():
    result = classify_retry_failure(
        chunks=[_chunk(0.3)],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=True),
    )
    assert result == RetrievalFailureCategory.INSUFFICIENT_EVIDENCE


def test_retry_collapses_llm_relevance_distinction():
    # The retry path does NOT consult llm_relevance: both LOW_SIMILARITY and
    # LOW_LLM_RELEVANCE collapse to INSUFFICIENT_EVIDENCE after a failed retry.
    relevant = classify_retry_failure(
        chunks=[_chunk(0.3)],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=True),
    )
    irrelevant = classify_retry_failure(
        chunks=[_chunk(0.3)],
        retrieval_grading=_grading(sufficient=False),
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
    )
    assert relevant == RetrievalFailureCategory.INSUFFICIENT_EVIDENCE
    assert irrelevant == RetrievalFailureCategory.INSUFFICIENT_EVIDENCE