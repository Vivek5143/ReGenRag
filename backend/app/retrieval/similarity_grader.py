"""Similarity-based retrieval grading (Phase 4, step 1).

Deterministically grades the retrieved chunks against a configurable cosine
similarity threshold. No external model or API is consulted: it reuses the
scores the retriever already computed (``RetrievedChunk.score``), where a
higher score means more relevant (cosine similarity = 1 - cosine distance).

LLM-based relevance grading, query rewriting, and re-retrieval are later
steps and deliberately live elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.vector_store import RetrievedChunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalGrading:
    """Verdict on whether the retrieved chunks hold sufficient evidence.

    ``sufficient`` is True when at least one retrieved chunk is relevant
    (``score >= threshold``). ``relevant_chunks`` and ``low_relevance_chunks``
    keep the full chunks so later self-healing steps can act on them.
    ``best_score`` and ``average_score`` cover all retrieved chunks and are
    ``None`` when there is no usable (finite) score, e.g. an empty retrieval.
    """

    sufficient: bool
    threshold: float
    relevant_chunks: list[RetrievedChunk] = field(default_factory=list)
    low_relevance_chunks: list[RetrievedChunk] = field(default_factory=list)
    best_score: float | None = None
    average_score: float | None = None
    reason: str = ""


def grade_retrieval(
    chunks: list[RetrievedChunk],
    threshold: float | None = None,
) -> RetrievalGrading:
    """Grade ``chunks`` against ``threshold`` (defaults to the configured one).

    A chunk is relevant when ``score >= threshold``. Retrieval is sufficient
    when at least one chunk is relevant; an empty result is insufficient but
    never raises. Non-finite scores (e.g. NaN from a degenerate query vector)
    never count as relevant and are excluded from the summary stats.
    """
    thr = (
        threshold if threshold is not None else settings.retrieval_similarity_threshold
    )

    relevant = [chunk for chunk in chunks if chunk.score >= thr]
    low_relevance = [chunk for chunk in chunks if not (chunk.score >= thr)]

    usable_scores = [chunk.score for chunk in chunks if math.isfinite(chunk.score)]
    best_score = max(usable_scores) if usable_scores else None
    average_score = (
        sum(usable_scores) / len(usable_scores) if usable_scores else None
    )

    if not chunks:
        reason = "no chunks retrieved"
    elif relevant:
        reason = (
            f"{len(relevant)} of {len(chunks)} chunks meet the similarity "
            f"threshold {thr}"
        )
    elif best_score is not None:
        reason = (
            f"no chunk meets the similarity threshold {thr} "
            f"(best score {best_score:.4f})"
        )
    else:
        reason = f"no chunk has a usable score against threshold {thr}"

    sufficient = bool(relevant)

    logger.info(
        "retrieval graded sufficient=%s threshold=%s relevant=%s total=%s",
        sufficient, thr, len(relevant), len(chunks),
    )

    return RetrievalGrading(
        sufficient=sufficient,
        threshold=thr,
        relevant_chunks=relevant,
        low_relevance_chunks=low_relevance,
        best_score=best_score,
        average_score=average_score,
        reason=reason,
    )
