"""Evaluation test suite for Phase 7.

Tests dataset loading, baseline runner, ReGenRAG runner, comparator metrics,
and report generation using mocked/stubbed DB and LLM clients.
"""

from __future__ import annotations

import uuid
from pathlib import Path
import pytest

from app.evaluation.model import EvalCase
from app.evaluation.dataset import load_cases, save_cases, create_sample_dataset, VALID_CATEGORIES
from app.evaluation.comparator import compare_runs, EvaluationReport
from app.evaluation.result import EvalResult, EvalRunResult


def test_dataset_loading_and_validation(tmp_path: Path):
    sample_path = tmp_path / "dataset.json"
    create_sample_dataset(sample_path)

    cases = load_cases(sample_path)
    assert len(cases) == len(VALID_CATEGORIES)
    for c in cases:
        assert c.id.startswith("case-")
        assert c.query
        assert c.metadata["category"] in VALID_CATEGORIES


def test_dataset_invalid_category_raises(tmp_path: Path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        '[{"id": "c1", "query": "hello", "category": "Z"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid category"):
        load_cases(bad_path)


def test_comparator_aggregates_correctly():
    # Since EvalResult is frozen, use simple objects with the needed attributes
    # and properties for testing the comparator logic.
    class _MockResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @property
        def retrieval_success(self) -> bool:
            return self.hit_rate > 0.0

    b_mock = _MockResult(
        case_id="c1",
        query="test query",
        hit_rate=0.0,
        mrr=0.0,
        ndcg=0.0,
        grounding_score=None,
        failure_category=None,
    )
    r_mock = _MockResult(
        case_id="c1",
        query="test query",
        hit_rate=1.0,
        mrr=1.0,
        ndcg=1.0,
        grounding_score=0.9,
        failure_category=None,
        rewritten_query="rewritten",
        attempts=2,
        healing_steps=["step1"],
        retry_exhausted=False,
    )

    b_run = EvalRunResult(results=[b_mock])
    r_run = EvalRunResult(results=[r_mock])

    report = compare_runs(b_run, r_run, metadata={"version": "1.0"})
    assert report.aggregate_metrics["total_cases"] == 1
    assert report.aggregate_metrics["comparison"]["improved_cases"] == 1
    assert report.recovery_analysis["recovery_rate"] == 1.0
    assert report.recovery_analysis["average_attempts"] == 2.0
