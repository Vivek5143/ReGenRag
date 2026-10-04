"""Evaluation layer.

See ``app.evaluation.model``, ``app.evaluation.metrics``,
``app.evaluation.result``, and ``app.evaluation.evaluator`` for the public API.
"""

from app.evaluation.evaluator import AnswerEvaluator, evaluate_case, evaluate_run
from app.evaluation.model import EvalCase
from app.evaluation.metrics import (
    compute_retrieval_metrics,
    hit_rate,
    mean_reciprocal_rank,
    normalized_dcg,
)
from app.evaluation.grounding import GroundingResult, grade_grounding
from app.evaluation.correctness import CorrectnessResult, grade_correctness
from app.evaluation.result import EvalResult, EvalRunResult

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
]
