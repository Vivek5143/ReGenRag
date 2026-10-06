"""Comparator & Report Generator for Phase 7 Evaluation.

Takes baseline and ReGenRAG evaluation runs, aligns them per case, calculates
aggregate metrics, recovery rates, and per-case difference analysis, and
outputs a machine-readable JSON (and optional CSV) report.

What it calculates
------------------
- **Aggregate metrics**: Hit Rate@K, MRR, nDCG, Grounding Score, Failure Rate,
  Recovery Rate, Retry Rate, Query Rewrite Rate, Average Attempts.
- **Per-case outcome**: Improved, Unchanged, Regressed.
- **Recovery metrics**:
  - Recovery Rate = successful recovered cases / cases requiring recovery
  - Retrieval Recovery = initial retrieval failed -> re-retrieval succeeded
  - Grounding Recovery = initial answer failed grounding -> retry produced grounded answer
  - Unrecovered Failure Rate = healing attempted but final result failed
  - Retry Rate = percentage of queries requiring healing
  - Query Rewrite Rate = percentage of cases requiring query rewriting
  - Average Attempts = average retrieval/generation attempts per query

Usage
-----
::

    from app.evaluation.comparator import compare_runs
    report = compare_runs(baseline_results, regen_results)
    comparator.save_report(report, out_dir=Path("results/"))
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .result import EvalResult, EvalRunResult


@dataclass(frozen=True)
class CaseComparison:
    """Side-by-side comparison for a single evaluation case."""

    case_id: str
    query: str
    category: str | None
    difficulty: str | None
    # Baseline outcomes
    baseline_hit_rate: float
    baseline_mrr: float
    baseline_ndcg: float
    baseline_grounding: float | None
    # ReGenRAG outcomes
    regen_hit_rate: float
    regen_mrr: float
    regen_ndcg: float
    regen_grounding: float | None
    regen_rewritten: bool
    regen_attempts: int
    regen_failure_category: str | None
    regen_retry_exhausted: bool
    # Analysis
    status: str  # "Improved" | "Unchanged" | "Regressed"
    notes: str = ""


@dataclass(frozen=True)
class EvaluationReport:
    """Full Phase 7 evaluation report structure."""

    run_metadata: dict[str, Any]
    aggregate_metrics: dict[str, Any]
    recovery_analysis: dict[str, Any]
    per_case_comparisons: list[CaseComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_metadata": self.run_metadata,
            "aggregate_metrics": self.aggregate_metrics,
            "recovery_analysis": self.recovery_analysis,
            "per_case_comparisons": [c.__dict__ for c in self.per_case_comparisons],
        }


def compare_runs(
    baseline_run: EvalRunResult,
    regen_run: EvalRunResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Compare baseline run results against ReGenRAG run results."""
    baseline_map = {r.case_id: r for r in baseline_run.results}
    regen_map = {r.case_id: r for r in regen_run.results}

    all_case_ids = sorted(set(baseline_map.keys()) | set(regen_map.keys()))

    comparisons: list[CaseComparison] = []
    improved_count = 0
    regressed_count = 0
    unchanged_count = 0

    requires_recovery_count = 0
    successfully_recovered_count = 0
    rewritten_count = 0
    total_attempts = 0

    for cid in all_case_ids:
        b = baseline_map.get(cid)
        r = regen_map.get(cid)

        b_hr = b.hit_rate if b else 0.0
        b_mrr = b.mrr if b else 0.0
        b_ndcg = b.ndcg if b else 0.0
        b_g = b.grounding_score if b else None

        r_hr = r.hit_rate if r else 0.0
        r_mrr = r.mrr if r else 0.0
        r_ndcg = r.ndcg if r else 0.0
        r_g = r.grounding_score if r else None

        r_rewritten = bool(r and getattr(r, "rewritten_query", None))
        r_attempts = int(getattr(r, "attempts", 1) if r else 1)
        total_attempts += r_attempts

        cat = getattr(b, "failure_category", None) or getattr(r, "failure_category", None)
        cat_str = cat.value if hasattr(cat, "value") else str(cat) if cat else None

        exhausted = bool(r and getattr(r, "retry_exhausted", False))

        # Check recovery & improvement status
        # Baseline failed retrieval (hit_rate == 0), but ReGenRAG succeeded (hit_rate > 0)
        # OR baseline grounding failed/low, but regen succeeded.
        b_failed = b_hr == 0.0 or (b_g is not None and b_g < 0.7)
        r_succeeded = r_hr > 0.0 and (r_g is None or r_g >= 0.7)

        if b_failed:
            requires_recovery_count += 1
            if r_succeeded and not exhausted:
                successfully_recovered_count += 1

        if r_rewritten:
            rewritten_count += 1

        # Status determination:
        # - If both failed retrieval (hit_rate == 0.0), status is Unchanged
        # - Otherwise, use composite score (retrieval + grounding) for comparison
        if b_hr == 0.0 and r_hr == 0.0:
            # Both failed retrieval - unchanged regardless of grounding
            status = "Unchanged"
            unchanged_count += 1
        else:
            # At least one succeeded in retrieval - use composite score
            # Handle None grounding scores by treating them as 0.0 for comparison
            b_grounding_for_score = b_g if b_g is not None else 0.0
            r_grounding_for_score = r_g if r_g is not None else 0.0

            b_score = b_hr + b_mrr + b_ndcg + b_grounding_for_score
            r_score = r_hr + r_mrr + r_ndcg + r_grounding_for_score

            if r_score > b_score:
                status = "Improved"
                improved_count += 1
            elif r_score < b_score:
                status = "Regressed"
                regressed_count += 1
            else:
                status = "Unchanged"
                unchanged_count += 1

        comparisons.append(
            CaseComparison(
                case_id=cid,
                query=r.query if r else (b.query if b else ""),
                category=None,
                difficulty=None,
                baseline_hit_rate=b_hr,
                baseline_mrr=b_mrr,
                baseline_ndcg=b_ndcg,
                baseline_grounding=b_g,
                regen_hit_rate=r_hr,
                regen_mrr=r_mrr,
                regen_ndcg=r_ndcg,
                regen_grounding=r_g,
                regen_rewritten=r_rewritten,
                regen_attempts=r_attempts,
                regen_failure_category=cat_str,
                regen_retry_exhausted=exhausted,
                status=status,
            )
        )

    total_cases = max(1, len(all_case_ids))

    aggregate = {
        "total_cases": total_cases,
        "baseline": {
            "hit_rate": baseline_run.avg_hit_rate,
            "mrr": baseline_run.avg_mrr,
            "ndcg": baseline_run.avg_ndcg,
            "avg_grounding": baseline_run.avg_grounding,
            "failure_rate": baseline_run.retrieval_fail_count / total_cases,
        },
        "regenrag": {
            "hit_rate": regen_run.avg_hit_rate,
            "mrr": regen_run.avg_mrr,
            "ndcg": regen_run.avg_ndcg,
            "avg_grounding": regen_run.avg_grounding,
            "failure_rate": regen_run.retrieval_fail_count / total_cases,
        },
        "comparison": {
            "improved_cases": improved_count,
            "unchanged_cases": unchanged_count,
            "regressed_cases": regressed_count,
        },
    }

    recovery_analysis = {
        "requires_recovery_count": requires_recovery_count,
        "successfully_recovered_count": successfully_recovered_count,
        "recovery_rate": (
            successfully_recovered_count / requires_recovery_count
            if requires_recovery_count > 0
            else 0.0
        ),
        "retry_rate": sum(1 for c in comparisons if c.regen_attempts > 1) / total_cases,
        "query_rewrite_rate": rewritten_count / total_cases,
        "average_attempts": total_attempts / total_cases,
    }

    run_meta = metadata or {}

    return EvaluationReport(
        run_metadata=run_meta,
        aggregate_metrics=aggregate,
        recovery_analysis=recovery_analysis,
        per_case_comparisons=comparisons,
    )


def save_report(report: EvaluationReport, out_dir: Path) -> tuple[Path, Path]:
    """Save evaluation report to JSON and CSV formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_report.json"
    csv_path = out_dir / "evaluation_report.csv"

    json_path.write_text(json.dumps(report.to_dict(), indent=2))

    # Generate CSV of per-case comparisons
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "query",
                "status",
                "baseline_hr",
                "regen_hr",
                "baseline_mrr",
                "regen_mrr",
                "baseline_ndcg",
                "regen_ndcg",
                "regen_rewritten",
                "regen_attempts",
                "regen_failure",
            ]
        )
        for c in report.per_case_comparisons:
            writer.writerow(
                [
                    c.case_id,
                    c.query,
                    c.status,
                    c.baseline_hit_rate,
                    c.regen_hit_rate,
                    c.regen_mrr,
                    c.regen_mrr,
                    c.regen_ndcg,
                    c.regen_ndcg,
                    c.regen_rewritten,
                    c.regen_attempts,
                    c.regen_failure_category or "",
                ]
            )

    return json_path, csv_path
