"""Tests for the metrics wrapper."""

import pytest
from app.evaluation.metrics_wrapper import per_case_metrics, aggregate_metrics, wrap_run
from app.evaluation.result import EvalResult, EvalRunResult


def test_per_case_metrics():
    """Test per_case_metrics returns correct structure."""
    results = [
        EvalResult(case_id="c1", query="q1", hit_rate=1.0, mrr=1.0, ndcg=1.0),
        EvalResult(case_id="c2", query="q2", hit_rate=0.5, mrr=0.5, ndcg=0.5),
    ]

    metrics = per_case_metrics(results)

    assert len(metrics) == 2
    assert metrics[0] == {"hit_rate": 1.0, "mrr": 1.0, "ndcg": 1.0}
    assert metrics[1] == {"hit_rate": 0.5, "mrr": 0.5, "ndcg": 0.5}


def test_aggregate_metrics():
    """Test aggregate_metrics computes correct averages."""
    results = [
        EvalResult(case_id="c1", query="q1", hit_rate=1.0, mrr=1.0, ndcg=1.0),
        EvalResult(case_id="c2", query="q2", hit_rate=0.0, mrr=0.0, ndcg=0.0),
    ]
    run = EvalRunResult(results=results)

    agg = aggregate_metrics(run)

    assert agg["avg_hit_rate"] == 0.5
    assert agg["avg_mrr"] == 0.5
    assert agg["avg_ndcg"] == 0.5


def test_wrap_run():
    """Test wrap_run returns per_case and aggregate."""
    results = [
        EvalResult(case_id="c1", query="q1", hit_rate=1.0, mrr=1.0, ndcg=1.0),
        EvalResult(case_id="c2", query="q2", hit_rate=0.0, mrr=0.0, ndcg=0.0),
    ]

    wrapped = wrap_run(results)

    assert "per_case" in wrapped
    assert "aggregate" in wrapped
    assert len(wrapped["per_case"]) == 2
    assert wrapped["aggregate"]["avg_hit_rate"] == 0.5


def test_empty_results():
    """Test with empty results list."""
    run = EvalRunResult(results=[])
    metrics = per_case_metrics([])
    agg = aggregate_metrics(run)

    assert metrics == []
    assert agg["avg_hit_rate"] == 0.0
    assert agg["avg_mrr"] == 0.0
    assert agg["avg_ndcg"] == 0.0


def test_deterministic_values():
    """Test with exact deterministic values for Hit Rate, MRR, nDCG."""
    # Case where first result is relevant (rank 1)
    results = [
        EvalResult(case_id="c1", query="q1", hit_rate=1.0, mrr=1.0, ndcg=1.0),
        # Case where relevant is at rank 2
        EvalResult(case_id="c2", query="q2", hit_rate=1.0, mrr=0.5, ndcg=0.6131471927654584),
        # Case where no relevant results
        EvalResult(case_id="c3", query="q3", hit_rate=0.0, mrr=0.0, ndcg=0.0),
    ]
    run = EvalRunResult(results=results)

    agg = aggregate_metrics(run)

    assert agg["avg_hit_rate"] == pytest.approx(2/3)
    assert agg["avg_mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert agg["avg_ndcg"] == pytest.approx((1.0 + 0.6131471927654584 + 0.0) / 3)
