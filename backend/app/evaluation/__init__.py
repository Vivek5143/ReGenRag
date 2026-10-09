"""Evaluation layer.

See ``app.evaluation.model``, ``app.evaluation.metrics``,
``app.evaluation.result``, and ``app.evaluation.evaluator`` for the public API.
"""
from .evaluator import (
    AnswerEvaluator,
    evaluate_case,
    evaluate_run,
)
from .model import EvalCase
from .metrics import (
    compute_retrieval_metrics,
    hit_rate,
    mean_reciprocal_rank,
    normalized_dcg,
)
from .grounding import GroundingResult, grade_grounding
from .correctness import CorrectnessResult, grade_correctness
from .result import EvalResult, EvalRunResult
from .baseline_runner import run_baseline
from .dataset import load_cases, create_sample_dataset
from .comparator import compare_runs, save_report, EvaluationReport

# We avoid importing regenrag_runner at the top level to prevent circular imports.
# It will be made available via __getattr__.

__all__ = [
    # models
    "EvalCase",
    "EvalResult",
    "EvalRunResult",
    # metrics
    "compute_retrieval_metrics",
    "hit_rate",
    "mean_reciprocal_rank",
    "normalized_dcg",
    # grounding
    "GroundingResult",
    "grade_grounding",
    # correctness
    "CorrectnessResult",
    "grade_correctness",
    # evaluator
    "AnswerEvaluator",
    "evaluate_case",
    "evaluate_run",
    # Phase 7 runners
    "run_baseline",
    "run_regenrag",  # made available via __getattr__
    # dataset
    "load_cases",
    "create_sample_dataset",
    # comparator / report
    "compare_runs",
    "save_report",
    "EvaluationReport",
]


def __getattr__(name):
    if name == "run_regenrag":
        from .regenrag_runner import run_regenrag
        return run_regenrag
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
