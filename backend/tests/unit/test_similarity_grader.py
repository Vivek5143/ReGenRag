"""Tests for the similarity-based retrieval grader.

Pure logic over in-memory :class:`RetrievedChunk` objects — no database
queries, no embeddings, no network, no external LLM. Verifies the sufficiency
rule, the threshold boundary, empty retrieval, and that the configured
threshold is actually used.
"""

import math
import uuid

import pytest

from app.core.config import settings
from app.retrieval.similarity_grader import RetrievalGrading, grade_retrieval
from app.retrieval.vector_store import RetrievedChunk

THRESHOLD = 0.65


def _chunk(score: float, *, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=index,
        page_number=1,
        content="evidence",
        score=score,
    )


def _chunks(*scores: float) -> list[RetrievedChunk]:
    return [_chunk(score, index=i) for i, score in enumerate(scores)]


def test_sufficient_retrieval():
    grading = grade_retrieval(_chunks(0.9, 0.8, 0.7), threshold=THRESHOLD)

    assert isinstance(grading, RetrievalGrading)
    assert grading.sufficient is True
    assert [c.score for c in grading.relevant_chunks] == [0.9, 0.8, 0.7]
    assert grading.low_relevance_chunks == []
    assert grading.best_score == pytest.approx(0.9)
    assert grading.average_score == pytest.approx(0.8)


def test_insufficient_retrieval():
    grading = grade_retrieval(_chunks(0.4, 0.3, 0.2), threshold=THRESHOLD)

    assert grading.sufficient is False
    assert grading.relevant_chunks == []
    assert [c.score for c in grading.low_relevance_chunks] == [0.4, 0.3, 0.2]
    assert grading.best_score == pytest.approx(0.4)
    assert "no chunk meets the similarity threshold" in grading.reason


def test_mixed_retrieval_follows_grading_rule():
    # Rule: relevant iff score >= threshold; sufficient iff any chunk relevant.
    grading = grade_retrieval(_chunks(0.9, 0.4, 0.7), threshold=THRESHOLD)

    assert grading.sufficient is True
    assert [c.score for c in grading.relevant_chunks] == [0.9, 0.7]
    assert [c.score for c in grading.low_relevance_chunks] == [0.4]
    assert "2 of 3 chunks" in grading.reason


def test_score_exactly_at_threshold_is_relevant():
    grading = grade_retrieval(_chunks(THRESHOLD, 0.3), threshold=THRESHOLD)

    assert grading.sufficient is True
    assert [c.score for c in grading.relevant_chunks] == [THRESHOLD]
    assert [c.score for c in grading.low_relevance_chunks] == [0.3]


def test_score_just_below_threshold_is_low_relevance():
    grading = grade_retrieval(_chunks(THRESHOLD - 1e-9), threshold=THRESHOLD)

    assert grading.sufficient is False
    assert grading.relevant_chunks == []
    assert len(grading.low_relevance_chunks) == 1


def test_empty_retrieval_is_insufficient_and_safe():
    grading = grade_retrieval([], threshold=THRESHOLD)

    assert grading.sufficient is False
    assert grading.relevant_chunks == []
    assert grading.low_relevance_chunks == []
    assert grading.best_score is None
    assert grading.average_score is None
    assert grading.reason == "no chunks retrieved"


def test_non_finite_scores_are_never_relevant():
    # Degenerate query vectors can yield NaN scores; they must not count as
    # relevant nor break the summary stats.
    grading = grade_retrieval(_chunks(math.nan, 0.9), threshold=THRESHOLD)

    assert grading.sufficient is True
    assert [c.score for c in grading.relevant_chunks] == [0.9]
    assert len(grading.low_relevance_chunks) == 1
    assert grading.best_score == pytest.approx(0.9)


def test_configured_threshold_is_used(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_similarity_threshold", 0.9)

    grading = grade_retrieval(_chunks(0.8))

    assert grading.sufficient is False
    assert grading.threshold == 0.9


def test_default_threshold_comes_from_settings():
    grading = grade_retrieval(_chunks(0.7))

    assert grading.threshold == settings.retrieval_similarity_threshold
    assert grading.sufficient is (
        0.7 >= settings.retrieval_similarity_threshold
    )


def test_explicit_threshold_overrides_config(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_similarity_threshold", 0.3)

    grading = grade_retrieval(_chunks(0.5), threshold=0.8)

    assert grading.sufficient is False
    assert grading.threshold == 0.8
