"""Evaluator: drive one case through retrieval + compute metrics.

This module is the single entry point for Phase-5 evaluation. It does **not**
call an LLM by default — it only performs retrieval (which in a real offline
run would be against a pre-populated session), then computes deterministic
retrieval metrics and assembles an :class:`.result.EvalResult`.

Two modes are supported:

- **``evaluate_case``** — evaluates a single case against a fixed set of
  retrieved chunks and metadata. Used by unit tests (no DB required).
  When an ``answer_evaluator`` is provided, it also runs LLM-based grounding
  and correctness checks.
- **``evaluate_run``** — aggregates per-case results into an
  :class:`.EvalRunResult`.

Both return an :class:`.result.EvalResult`. Callers can wrap either in a
loop over a list of cases and aggregate with :class:`.result.EvalRunResult`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from ..core.logging import get_logger
from .correctness import CorrectnessResult, grade_correctness
from .grounding import GroundingResult, grade_grounding
from .metrics import compute_retrieval_metrics
from .model import EvalCase
from .result import EvalResult, EvalRunResult
from ..generation import llm
from ..retrieval.vector_store import RetrievedChunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class AnswerEvaluator:
    """Optional LLM-based evaluator for answer quality.

    Pass an instance of this to :func:`evaluate_case` to enable grounding
    and correctness evaluation. Without it, only deterministic retrieval
    metrics are computed.
    """

    client: llm.LLMClient | None = None


def evaluate_case(
    case: EvalCase,
    retrieved: list[RetrievedChunk],
    *,
    answer: str | None = None,
    evidence: str | None = None,
    failure_category=None,
    rewritten_query: str | None = None,
    k: int | None = None,
    answer_evaluator: AnswerEvaluator | None = None,
) -> EvalResult:
    """Evaluate a single case against pre-fetched retrieval results.

    ``retrieved`` is the ranked list of chunks returned for ``case.query``.
    ``case.relevant_chunk_ids`` is the oracle ground truth. Returns an
    :class:`EvalResult` with hit rate, MRR, and nDCG populated.

    If ``answer_evaluator`` is provided along with ``answer`` and ``evidence``,
    LLM-based grounding and correctness scores are also computed and included
    in the result. Without the evaluator, those fields remain ``None``.

    ``failure_category`` and ``rewritten_query`` are carried through from the
    RAG service so the evaluator can record whether this case triggered any
    self-healing path.
    """

    relevant_ids: Iterable[str | uuid.UUID] | None = (
        case.relevant_chunk_ids if case.relevant_chunk_ids is not None else []
    )

    metrics = compute_retrieval_metrics(retrieved, relevant_ids, k=k)

    # --- LLM-based answer evaluation (optional) ---
    grounding_score: float | None = None
    answer_quality: float | None = None

    if answer_evaluator is not None and answer is not None:
        # Grounding: is the answer faithful to the evidence?
        if evidence:
            try:
                gr = grade_grounding(
                    case.query, evidence, answer,
                    client=answer_evaluator.client,
                )
                grounding_score = gr.score
            except Exception as exc:
                logger.warning("grounding eval failed case=%s error=%s", case.id, exc)
                grounding_score = None
        else:
            grounding_score = 0.0  # no evidence → no grounding possible

        # Correctness: does the answer match the expected answer?
        if case.expected_answer and answer:
            try:
                cr = grade_correctness(
                    case.query, case.expected_answer, answer,
                    client=answer_evaluator.client,
                )
                answer_quality = cr.score
            except Exception as exc:
                logger.warning("correctness eval failed case=%s error=%s", case.id, exc)
                answer_quality = None

    return EvalResult(
        case_id=case.id,
        query=case.query,
        hit_rate=metrics["hit_rate"],
        mrr=metrics["mrr"],
        ndcg=metrics["ndcg"],
        grounding_score=grounding_score,
        answer_quality=answer_quality,
        failure_category=failure_category,
        rewritten_query=rewritten_query,
        expected_answer=case.expected_answer,
    )


def evaluate_run(
    cases: list[EvalCase],
    case_results: dict[str, EvalResult],
) -> EvalRunResult:
    """Aggregate per-case results into an :class:`EvalRunResult`.

    ``case_results`` is keyed by ``case.id``. Missing cases are skipped
    silently — they simply don't contribute to the aggregate.
    """

    ordered = [
        case_results[c.id]
        for c in cases
        if c.id in case_results
    ]
    return EvalRunResult(results=ordered)
