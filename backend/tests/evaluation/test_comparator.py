"""Tests for the comparator."""

import pytest
import json
from pathlib import Path
from app.evaluation.comparator import compare_runs, EvaluationReport, save_report
from app.evaluation.result import EvalResult, EvalRunResult


def test_compare_runs_structure():
    """Test that compare_runs returns correct structure."""
    # Create mock results
    baseline_results = [
        EvalResult(
            case_id="case-001",
            query="Test query 1",
            hit_rate=0.0,
            mrr=0.0,
            ndcg=0.0,
            grounding_score=None,
            expected_answer="Expected answer 1",
        ),
        EvalResult(
            case_id="case-002",
            query="Test query 2",
            hit_rate=1.0,
            mrr=1.0,
            ndcg=1.0,
            grounding_score=0.8,
            expected_answer="Expected answer 2",
        ),
    ]

    regen_results = [
        EvalResult(
            case_id="case-001",
            query="Test query 1",
            hit_rate=1.0,
            mrr=1.0,
            ndcg=1.0,
            grounding_score=0.9,
            rewritten_query="Rewritten query 1",
            attempts=2,
            healing_steps=["step1"],
            retry_exhausted=False,
            expected_answer="Expected answer 1",
        ),
        EvalResult(
            case_id="case-002",
            query="Test query 2",
            hit_rate=1.0,
            mrr=1.0,
            ndcg=1.0,
            grounding_score=0.85,
            rewritten_query=None,
            attempts=1,
            healing_steps=[],
            retry_exhausted=False,
            expected_answer="Expected answer 2",
        ),
    ]

    baseline_run = EvalRunResult(results=baseline_results)
    regen_run = EvalRunResult(results=regen_results)

    report = compare_runs(baseline_run, regen_run, metadata={"version": "1.0"})

    # Check structure
    assert isinstance(report, EvaluationReport)
    assert report.run_metadata["version"] == "1.0"
    assert report.aggregate_metrics["total_cases"] == 2
    assert len(report.per_case_comparisons) == 2

    # Check aggregate metrics
    assert report.aggregate_metrics["baseline"]["hit_rate"] == 0.5
    assert report.aggregate_metrics["regenrag"]["hit_rate"] == 1.0
    assert report.aggregate_metrics["baseline"]["mrr"] == 0.5
    assert report.aggregate_metrics["regenrag"]["mrr"] == 1.0
    assert report.aggregate_metrics["baseline"]["ndcg"] == 0.5
    assert report.aggregate_metrics["regenrag"]["ndcg"] == 1.0

    # Check recovery analysis
    assert report.recovery_analysis["requires_recovery_count"] == 1  # case-001 failed baseline
    assert report.recovery_analysis["successfully_recovered_count"] == 1  # case-001 recovered
    assert report.recovery_analysis["recovery_rate"] == 1.0
    assert report.recovery_analysis["retry_rate"] == 0.5  # 1 out of 2 cases had retries (attempts > 1)
    assert report.recovery_analysis["query_rewrite_rate"] == 0.5  # 1 out of 2 cases rewritten
    assert report.recovery_analysis["average_attempts"] == 1.5  # (1 + 2) / 2

    # Check per-case comparisons
    comparisons = report.per_case_comparisons
    assert len(comparisons) == 2

    # First case: Improved
    assert comparisons[0].case_id == "case-001"
    assert comparisons[0].status == "Improved"
    assert comparisons[0].baseline_hit_rate == 0.0
    assert comparisons[0].regen_hit_rate == 1.0
    assert comparisons[0].regen_rewritten == True
    assert comparisons[0].regen_attempts == 2

    # Second case: Improved (grounding improved from 0.8 to 0.85)
    assert comparisons[1].case_id == "case-002"
    assert comparisons[1].status == "Improved"
    assert comparisons[1].baseline_hit_rate == 1.0
    assert comparisons[1].regen_hit_rate == 1.0
    assert comparisons[1].regen_rewritten == False
    assert comparisons[1].regen_attempts == 1


def test_save_report(tmp_path: Path):
    """Test saving report to JSON and CSV."""
    baseline_results = [
        EvalResult(
            case_id="case-001",
            query="Test query",
            hit_rate=0.0,
            mrr=0.0,
            ndcg=0.0,
            grounding_score=None,
            expected_answer="Expected answer",
        )
    ]

    regen_results = [
        EvalResult(
            case_id="case-001",
            query="Test query",
            hit_rate=1.0,
            mrr=1.0,
            ndcg=1.0,
            grounding_score=0.9,
            rewritten_query="Rewritten query",
            attempts=2,
            healing_steps=["step1"],
            retry_exhausted=False,
            expected_answer="Expected answer",
        )
    ]

    baseline_run = EvalRunResult(results=baseline_results)
    regen_run = EvalRunResult(results=regen_results)

    report = compare_runs(baseline_run, regen_run)

    json_path, csv_path = save_report(report, tmp_path)

    # Check files exist
    assert json_path.exists()
    assert csv_path.exists()

    # Check JSON content
    json_data = json.loads(json_path.read_text())
    assert json_data["aggregate_metrics"]["total_cases"] == 1
    assert json_data["aggregate_metrics"]["comparison"]["improved_cases"] == 1
    assert len(json_data["per_case_comparisons"]) == 1

    # Check CSV content
    csv_content = csv_path.read_text()
    lines = csv_content.strip().split('\n')
    assert len(lines) == 2  # header + data
    assert "case_id" in lines[0]
    assert "case-001" in lines[1]
    assert "Improved" in lines[1]


def test_edge_cases():
    """Test edge cases like zero recovery cases."""
    baseline_results = [
        EvalResult(
            case_id="case-001",
            query="Test query",
            hit_rate=0.0,
            mrr=0.0,
            ndcg=0.0,
            grounding_score=None,
            expected_answer="Expected answer",
        )
    ]

    regen_results = [
        EvalResult(
            case_id="case-001",
            query="Test query",
            hit_rate=0.0,  # Still failed
            mrr=0.0,
            ndcg=0.0,
            grounding_score=0.5,  # Still low
            rewritten_query="Rewritten query",
            attempts=2,
            healing_steps=["step1"],
            retry_exhausted=True,  # Retries exhausted
            expected_answer="Expected answer",
        )
    ]

    baseline_run = EvalRunResult(results=baseline_results)
    regen_run = EvalRunResult(results=regen_results)

    report = compare_runs(baseline_run, regen_run)

    # Should require recovery but not succeed
    assert report.recovery_analysis["requires_recovery_count"] == 1
    assert report.recovery_analysis["successfully_recovered_count"] == 0
    assert report.recovery_analysis["recovery_rate"] == 0.0
    assert report.recovery_analysis["retry_rate"] == 1.0  # 1 out of 1 cases had retries
    assert report.recovery_analysis["query_rewrite_rate"] == 1.0  # 1 out of 1 cases rewritten
    assert report.per_case_comparisons[0].status == "Unchanged"  # Still failed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
