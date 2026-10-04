"""Retrieval failure classification for Phase 4.

This module classifies *why* evidence was insufficient, using only the
existing deterministic signals produced by the similarity and LLM relevance
graders.

Important:
- The pipeline's decision to generate an answer is controlled by the
  similarity grader (``RetrievalGrading.sufficient``).
- LLM relevance grading is treated as an informational signal for
  classification; it does not change answer-generation behavior in Phase 4.

The classifier exposes a small set of stable categories:
- NO_RESULTS: zero chunks were retrieved.
- LOW_SIMILARITY: chunks were retrieved, but similarity gating failed.
- LOW_LLM_RELEVANCE: chunks were retrieved, similarity gating failed, and
  the LLM judged that there was no relevant evidence.
- INSUFFICIENT_EVIDENCE: after the second retrieval attempt, similarity
  gating failed again while evidence existed.
"""

from __future__ import annotations

from enum import Enum

from app.retrieval.llm_grader import LlmRetrievalGrading
from app.retrieval.similarity_grader import RetrievalGrading
from app.retrieval.vector_store import RetrievedChunk


class RetrievalFailureCategory(str, Enum):
    """Stable classification labels for retrieval failures."""

    NO_RESULTS = "NO_RESULTS"
    LOW_SIMILARITY = "LOW_SIMILARITY"
    LOW_LLM_RELEVANCE = "LOW_LLM_RELEVANCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def classify_initial_failure(
    *,
    chunks: list[RetrievedChunk],
    retrieval_grading: RetrievalGrading,
    llm_relevance: LlmRetrievalGrading,
) -> RetrievalFailureCategory | None:
    """Classify the first (pre-rewrite) retrieval failure.

    Returns None when similarity gating deemed evidence sufficient.
    """

    if not chunks:
        return RetrievalFailureCategory.NO_RESULTS

    if retrieval_grading.sufficient:
        return None

    # Similarity gating failed; the LLM relevance signal refines the label.
    if llm_relevance.has_relevant_evidence:
        return RetrievalFailureCategory.LOW_SIMILARITY

    return RetrievalFailureCategory.LOW_LLM_RELEVANCE


def classify_retry_failure(
    *,
    chunks: list[RetrievedChunk],
    retrieval_grading: RetrievalGrading,
    llm_relevance: LlmRetrievalGrading,
) -> RetrievalFailureCategory | None:
    """Classify the second (post-rewrite) retrieval failure.

    Returns None when similarity gating deemed evidence sufficient.
    """

    if not chunks:
        return RetrievalFailureCategory.NO_RESULTS

    if retrieval_grading.sufficient:
        return None

    # At this point evidence still wasn't sufficient after the one allowed
    # remediation step.
    return RetrievalFailureCategory.INSUFFICIENT_EVIDENCE
