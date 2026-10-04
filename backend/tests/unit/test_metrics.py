"""Tests for deterministic retrieval metrics.

Covers Hit Rate, MRR, and nDCG against mocked :class:`RetrievedChunk`
objects. No database, no LLM, no network.
"""

from __future__ import annotations

import math
import uuid

import pytest

from app.evaluation.metrics import (
    compute_retrieval_metrics,
    hit_rate,
    mean_reciprocal_rank,
    normalized_dcg,
)
from app.retrieval.vector_store import RetrievedChunk


def _chunk(chunk_id: uuid.UUID, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content="evidence",
        score=score,
    )


def _ids(*chunk_ids: uuid.UUID) -> set[uuid.UUID]:
    return set(chunk_ids)


# ---------------------------------------------------------------------------
# Hit Rate
# ---------------------------------------------------------------------------


def test_hit_rate_when_relevant_exists():
    c = _chunk(uuid.uuid4())
    assert hit_rate([c], _ids(c.chunk_id)) == 1.0


def test_hit_rate_when_no_match():
    a, b = uuid.uuid4(), uuid.uuid4()
    assert hit_rate([_chunk(a)], _ids(b)) == 0.0


def test_hit_rate_empty_retrieval():
    assert hit_rate([], _ids(uuid.uuid4())) == 0.0


def test_hit_rate_empty_oracle():
    c = _chunk(uuid.uuid4())
    assert hit_rate([c], set()) == 0.0


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank
# ---------------------------------------------------------------------------


def test_mrr_rank_1():
    c = _chunk(uuid.uuid4())
    assert mean_reciprocal_rank([c], _ids(c.chunk_id)) == pytest.approx(1.0)


def test_mrr_rank_2():
    a, b = uuid.uuid4(), uuid.uuid4()
    assert mean_reciprocal_rank(
        [_chunk(a), _chunk(b)], _ids(b)
    ) == pytest.approx(0.5)


def test_mrr_no_match():
    a, b = uuid.uuid4(), uuid.uuid4()
    assert mean_reciprocal_rank([_chunk(a)], _ids(b)) == 0.0


def test_mrr_empty_retrieval():
    assert mean_reciprocal_rank([], _ids(uuid.uuid4())) == 0.0


# ---------------------------------------------------------------------------
# nDCG
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ranking_single():
    c = _chunk(uuid.uuid4())
    assert normalized_dcg([c], _ids(c.chunk_id)) == pytest.approx(1.0)


def test_ndcg_partial_relevance():
    a, b = uuid.uuid4(), uuid.uuid4()
    # ``a`` is relevant, ``b`` is not. DCG = 1/log2(2) + 0 = 1.0.
    # IDCG (ideal) = 1/log2(2) = 1.0. nDCG = 1.0.
    assert normalized_dcg([_chunk(a), _chunk(b)], _ids(a)) == pytest.approx(1.0)


def test_ndcg_irrelevant_first():
    a, b = uuid.uuid4(), uuid.uuid4()
    # Relevant chunk is at position 2.
    # DCG = 0/log2(2) + 1/log2(3) ≈ 0.6309
    # IDCG = 1/log2(2) = 1.0
    result = normalized_dcg([_chunk(b), _chunk(a)], _ids(a))
    expected = (1.0 / math.log2(3)) / 1.0
    assert result == pytest.approx(expected)


def test_ndcg_empty_retrieval():
    assert normalized_dcg([], _ids(uuid.uuid4())) == 0.0


def test_ndcg_empty_oracle():
    c = _chunk(uuid.uuid4())
    assert normalized_dcg([c], set()) == 0.0


def test_ndcg_k_cap():
    a, b = uuid.uuid4(), uuid.uuid4()
    c1 = _chunk(a)
    c2 = _chunk(uuid.uuid4())
    c3 = _chunk(b)
    # With k=1, we only see c1 (relevant). DCG = 1/log2(2) = 1.0.
    # IDCG with 1 slot and 2 relevant chunks = 1/log2(2) = 1.0. nDCG = 1.0.
    assert normalized_dcg([c1, c2, c3], _ids(a, b), k=1) == pytest.approx(1.0)


def test_ndcg_k_larger_than_list():
    c = _chunk(uuid.uuid4())
    # k=5 but only 1 chunk retrieved — behaves like k=1.
    assert normalized_dcg([c], _ids(c.chunk_id), k=5) == pytest.approx(1.0)


def test_ndcg_no_relevant_in_top_k():
    a, b = uuid.uuid4(), uuid.uuid4()
    # Relevant chunk is at position 2, k=1: nothing in the top-1.
    result = normalized_dcg([_chunk(b), _chunk(a)], _ids(a), k=1)
    assert result == 0.0


# ---------------------------------------------------------------------------
# compute_retrieval_metrics
# ---------------------------------------------------------------------------


def test_compute_returns_all_three_metrics():
    c = _chunk(uuid.uuid4())
    metrics = compute_retrieval_metrics([c], [c.chunk_id])
    assert "hit_rate" in metrics
    assert "mrr" in metrics
    assert "ndcg" in metrics


def test_compute_string_chunk_ids_accepted():
    c = _chunk(uuid.uuid4())
    metrics = compute_retrieval_metrics([c], [str(c.chunk_id)])
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg"] == pytest.approx(1.0)


def test_compute_empty_retrieval():
    metrics = compute_retrieval_metrics([], [uuid.uuid4()])
    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["ndcg"] == 0.0


def test_compute_invalid_type_raises():
    with pytest.raises(TypeError):
        compute_retrieval_metrics([], [123])  # type: ignore[arg-type]
