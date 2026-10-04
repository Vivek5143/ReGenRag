"""Tests for evaluation result models.

Verifies :class:`EvalResult` properties and :class:`EvalRunResult` aggregates.
Pure in-memory — no DB, no LLM, no network.
"""

from __future__ import annotations

import pytest

from app.evaluation.result import EvalResult, EvalRunResult
from app.retrieval.failure_classifier import RetrievalFailureCategory


def _result(
    *,
    case_id: str = "q1",
    query: str = "What is X?",
    hit_rate: float = 1.0,
    mrr: float = 1.0,
    ndcg: float = 1.0,
    failure_category=None,
    rewritten_query=None,
    expected_answer=None,
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        query=query,
        hit_rate=hit_rate,
        mrr=mrr,
        ndcg=ndcg,
        failure_category=failure_category,
        rewritten_query=rewritten_query,
        expected_answer=expected_answer,
    )


# ---------------------------------------------------------------------------
# EvalResult properties
# ---------------------------------------------------------------------------


def test_retrieval_success_when_hit_rate_positive():
    r = _result(hit_rate=1.0)
    assert r.retrieval_success is True


def test_retrieval_failure_when_hit_rate_zero():
    r = _result(hit_rate=0.0)
    assert r.retrieval_success is False


def test_reattempted_when_rewritten_query_present():
    r = _result(rewritten_query="rewritten Q")
    assert r.reattempted is True


def test_not_reattempted_when_no_rewrite():
    r = _result(rewritten_query=None)
    assert r.reattempted is False


# ---------------------------------------------------------------------------
# EvalRunResult aggregates
# ---------------------------------------------------------------------------


def test_empty_run_returns_zeros():
    run = EvalRunResult()
    assert run.count == 0
    assert run.avg_hit_rate == 0.0
    assert run.avg_mrr == 0.0
    assert run.avg_ndcg == 0.0
    assert run.retrieval_succeed_count == 0
    assert run.retrieval_fail_count == 0
    assert run.rewrite_rate == 0.0


def test_single_success_case():
    r = _result(hit_rate=1.0, mrr=1.0, ndcg=1.0)
    run = EvalRunResult(results=[r])
    assert run.count == 1
    assert run.avg_hit_rate == 1.0
    assert run.avg_mrr == 1.0
    assert run.avg_ndcg == 1.0
    assert run.retrieval_succeed_count == 1
    assert run.retrieval_fail_count == 0
    assert run.rewrite_rate == 0.0


def test_single_failure_case():
    r = _result(hit_rate=0.0, mrr=0.0, ndcg=0.0)
    run = EvalRunResult(results=[r])
    assert run.avg_hit_rate == 0.0
    assert run.retrieval_succeed_count == 0
    assert run.retrieval_fail_count == 1


def test_mixed_cases():
    success = _result(hit_rate=1.0, mrr=1.0, ndcg=1.0, case_id="s")
    failure = _result(hit_rate=0.0, mrr=0.0, ndcg=0.0, case_id="f")
    run = EvalRunResult(results=[success, failure])
    assert run.count == 2
    assert run.avg_hit_rate == pytest.approx(0.5)
    assert run.avg_mrr == pytest.approx(0.5)
    assert run.retrieval_succeed_count == 1
    assert run.retrieval_fail_count == 1


def test_rewrite_rate():
    rewritten = _result(rewritten_query="q'")
    no_rewrite = _result(rewritten_query=None)
    run = EvalRunResult(results=[rewritten, no_rewrite])
    assert run.rewrite_rate == pytest.approx(0.5)


def test_failure_categories():
    r1 = _result(failure_category=RetrievalFailureCategory.NO_RESULTS, case_id="a")
    r2 = _result(
        failure_category=RetrievalFailureCategory.LOW_SIMILARITY, case_id="b"
    )
    r3 = _result(failure_category=None, case_id="c")
    run = EvalRunResult(results=[r1, r2, r3])
    cats = run.failure_categories
    assert cats["NO_RESULTS"] == 1
    assert cats["LOW_SIMILARITY"] == 1
    assert cats["NONE"] == 1


def test_missing_case_skipped_in_evaluate_run():
    """If an EvalResult is missing from case_results, it is silently skipped."""
    from app.evaluation.model import EvalCase

    cases = [EvalCase(id="a", query="a"), EvalCase(id="b", query="b")]
    results = {"a": _result(case_id="a")}  # b is missing
    from app.evaluation.evaluator import evaluate_run

    run = evaluate_run(cases, results)
    assert run.count == 1
    assert run.results[0].case_id == "a"
