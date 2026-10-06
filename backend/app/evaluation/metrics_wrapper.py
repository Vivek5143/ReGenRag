"""Thin wrapper for evaluation metrics.

Provides helpers to compute per‑case metric dictionaries and aggregate
summaries using the existing deterministic retrieval metric functions.
"""

from __future__ import annotations

from typing import List, Dict

from .result import EvalResult, EvalRunResult
from .metrics import compute_retrieval_metrics


def per_case_metrics(results: List[EvalResult]) -> List[Dict[str, float]]:
    """Return a list of metric dicts for each ``EvalResult``.

    The dict contains ``hit_rate``, ``mrr`` and ``ndcg`` exactly as stored in the
    result. This wrapper is deliberately lightweight – it does not recompute the
    metrics from raw chunks because the ``EvalResult`` already holds the values.
    """
    return [{"hit_rate": r.hit_rate, "mrr": r.mrr, "ndcg": r.ndcg} for r in results]


def aggregate_metrics(run: EvalRunResult) -> Dict[str, float]:
    """Aggregate metrics across a run using the properties on ``EvalRunResult``.

    Returns ``avg_hit_rate``, ``avg_mrr`` and ``avg_ndcg``.
    """
    return {
        "avg_hit_rate": run.avg_hit_rate,
        "avg_mrr": run.avg_mrr,
        "avg_ndcg": run.avg_ndcg,
    }


def wrap_run(results: List[EvalResult]) -> Dict[str, object]:
    """Convenient wrapper returning per‑case and aggregate metrics.

    The returned structure matches what the CLI serialises to JSON:

    ``{"per_case": [...], "aggregate": {...}}``
    """
    run = EvalRunResult(results=results)
    return {"per_case": per_case_metrics(results), "aggregate": aggregate_metrics(run)}

__all__ = ["per_case_metrics", "aggregate_metrics", "wrap_run"]
