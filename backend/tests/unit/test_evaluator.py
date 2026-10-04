"""Tests for the evaluation entry point (evaluator module).

Verifies that ``evaluate_case`` correctly wires together an :class:`EvalCase`,
a list of :class:`RetrievedChunk`, and the resulting :class:`EvalResult`.
"""

from __future__ import annotations

import uuid

import pytest

from app.evaluation import evaluate_case, evaluate_run
from app.evaluation.model import EvalCase
from app.evaluation.result import EvalResult, EvalRunResult
from app.retrieval.failure_classifier import RetrievalFailureCategory
from app.retrieval.vector_store import RetrievedChunk


def _chunk(chunk_id: uuid.UUID, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content="evidence text",
        score=score,
    )


def test_evaluate_case_hits():
    """A relevant chunk in the retrieval produces a successful EvalResult."""
    chunk_id = uuid.uuid4()
    case = EvalCase(id="q1", query="What is X?", relevant_chunk_ids=[chunk_id])
    retrieved = [_chunk(chunk_id)]

    result = evaluate_case(case, retrieved)

    assert result.case_id == "q1"
    assert result.hit_rate == 1.0
    assert result.mrr == pytest.approx(1.0)
    assert result.ndcg == pytest.approx(1.0)
    assert result.failure_category is None
    assert result.rewritten_query is None


def test_evaluate_case_miss():
    """No relevant chunk -> zero metrics."""
    chunk_a = uuid.uuid4()
    chunk_b = uuid.uuid4()
    case = EvalCase(id="q1", query="What is X?", relevant_chunk_ids=[chunk_a])
    retrieved = [_chunk(chunk_b)]

    result = evaluate_case(case, retrieved)

    assert result.hit_rate == 0.0
    assert result.mrr == 0.0
    assert result.ndcg == 0.0


def test_evaluate_case_empty_retrieval():
    """Empty retrieval is safe and yields zeros."""
    case = EvalCase(id="q1", query="What is X?", relevant_chunk_ids=[uuid.uuid4()])
    result = evaluate_case(case, [])
    assert result.hit_rate == 0.0
    assert result.mrr == 0.0
    assert result.ndcg == 0.0


def test_evaluate_case_without_oracle():
    """When relevant_chunk_ids is None, all metrics are zero."""
    case = EvalCase(id="q1", query="What is X?")
    retrieved = [_chunk(uuid.uuid4())]
    result = evaluate_case(case, retrieved)
    assert result.hit_rate == 0.0
    assert result.mrr == 0.0
    assert result.ndcg == 0.0


def test_evaluate_case_carries_failure_category():
    case = EvalCase(id="q1", query="What is X?")
    result = evaluate_case(
        case,
        [],
        failure_category=RetrievalFailureCategory.NO_RESULTS,
        rewritten_query="rewritten",
    )
    assert result.failure_category == RetrievalFailureCategory.NO_RESULTS
    assert result.rewritten_query == "rewritten"
    assert result.retrieval_success is False
    assert result.reattempted is True


def test_evaluate_case_expected_answer_passed_through():
    case = EvalCase(id="q1", query="Q", expected_answer="A")
    result = evaluate_case(case, [])
    assert result.expected_answer == "A"


def test_evaluate_case_k_cap():
    """The ``k`` parameter restricts nDCG computation to the top-K.

    hit_rate and mrr are computed over the full retrieval list (they are not
    position-capped); only nDCG is affected by ``k``.
    """
    relevant = uuid.uuid4()
    irrelevant = uuid.uuid4()
    case = EvalCase(id="q1", query="Q", relevant_chunk_ids=[relevant])
    retrieved = [
        _chunk(irrelevant),  # rank 1, irrelevant
        _chunk(relevant),    # rank 2, relevant
    ]
    # With k=1, only the first chunk is scored in nDCG — hit_rate=1, mrr=0.5,
    # ndcg=0 (nothing relevant in the top-1).
    result = evaluate_case(case, retrieved, k=1)
    assert result.hit_rate == 1.0
    assert result.mrr == pytest.approx(0.5)
    assert result.ndcg == 0.0
    # With k=2, both chunks are scored. DCG = 0/log2(2) + 1/log2(3) ≈ 0.6309,
    # IDCG = 1/log2(2) = 1.0, so nDCG ≈ 0.6309 (relevant chunk is at rank 2).
    result2 = evaluate_case(case, retrieved, k=2)
    assert result2.hit_rate == 1.0
    assert result2.mrr == pytest.approx(0.5)
    assert result2.ndcg == pytest.approx(0.6309297535714575)


def test_evaluate_run_aggregates():
    chunk_a = uuid.UUID("11111111-1111-1111-1111-111111111111")
    chunk_b = uuid.UUID("22222222-2222-2222-2222-222222222222")
    case_a = EvalCase(id="a", query="A", relevant_chunk_ids=[chunk_a])
    case_b = EvalCase(id="b", query="B", relevant_chunk_ids=[chunk_b])

    result_a = evaluate_case(case_a, [_chunk(chunk_a)])
    result_b = evaluate_case(case_b, [])

    run = evaluate_run([case_a, case_b], {"a": result_a, "b": result_b})
    assert isinstance(run, EvalRunResult)
    assert run.count == 2
    assert run.avg_hit_rate == pytest.approx(0.5)
    assert run.avg_mrr == pytest.approx(0.5)
