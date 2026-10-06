"""Evaluation result model.

A single evaluation run produces one :class:`EvalResult`. Results are
aggregated from per-case :class:`.model.EvalCase` evaluations and, when a
collection is run, from per-case aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Any

from app.retrieval.failure_classifier import RetrievalFailureCategory


@dataclass(frozen=True)
class EvalResult:
    """Result of evaluating one :class:`.model.EvalCase` against a RAG answer.

    Fields marked "future" are reserved slots; they remain ``None`` until an
    LLM-based or reference-answer metric is implemented. This keeps the
    result schema extensible without breaking consumers.
    """

    case_id: str
    query: str
    # Retrieval
    hit_rate: float
    mrr: float
    ndcg: float
    # Grounding / faithfulness (Phase 5)
    grounding_score: float | None = None
    # Answer correctness (Phase 5)
    answer_quality: float | None = None
    # ReGenRAG recovery signals
    failure_category: RetrievalFailureCategory | None = None
    rewritten_query: str | None = None
    # Human-authored expected answer, echoed back for audit (not used for
    # scoring here — answer_quality is kept as a future float placeholder).
    expected_answer: str | None = None
    # Healing metadata (Phase 6)
    attempts: int = 1
    healing_steps: List[Any] = field(default_factory=list)
    retry_exhausted: bool = False

    @property
    def retrieval_success(self) -> bool:
        """True when at least one relevant chunk was retrieved."""
        return self.hit_rate > 0.0

    @property
    def reattempted(self) -> bool:
        """True when the query was rewritten and re-retrieved."""
        return self.rewritten_query is not None


@dataclass(frozen=True)
class EvalRunResult:
    """Aggregated results across multiple :class:`EvalResult` cases."""

    results: List[EvalResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def avg_hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.hit_rate for r in self.results) / self.count

    @property
    def avg_mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.mrr for r in self.results) / self.count

    @property
    def avg_ndcg(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.ndcg for r in self.results) / self.count

    @property
    def retrieval_succeed_count(self) -> int:
        return sum(1 for r in self.results if r.retrieval_success)

    @property
    def retrieval_fail_count(self) -> int:
        return self.count - self.retrieval_succeed_count

    @property
    def rewrite_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.reattempted) / self.count

    @property
    def failure_categories(self) -> dict[str, int]:
        buckets: dict[str, int] = {}
        for r in self.results:
            if r.failure_category is None:
                key = "NONE"
            else:
                key = r.failure_category.value
            buckets[key] = buckets.get(key, 0) + 1
        return buckets

    @property
    def avg_grounding(self) -> float:
        """Average grounding score over cases that have one."""
        scores = [r.grounding_score for r in self.results if r.grounding_score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_answer_quality(self) -> float:
        """Average answer correctness over cases that have an expected answer."""
        scores = [r.answer_quality for r in self.results if r.answer_quality is not None]
        return sum(scores) / len(scores) if scores else 0.0
