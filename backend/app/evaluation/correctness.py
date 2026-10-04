"""LLM-based answer correctness evaluator (Phase 5).

Asks the LLM to judge whether a generated answer is semantically
correct compared to an expected reference answer. Returns a normalized
score (0.0–1.0) plus a one-sentence explanation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.generation import llm
from app.evaluation.prompts import build_correctness_prompt

logger = get_logger(__name__)


@dataclass(frozen=True)
class CorrectnessResult:
    """Single correctness judgement result."""

    score: float
    reason: str | None = None
    parse_failed: bool = False


def _extract_json_object(text: str) -> dict | None:
    """Return the balanced JSON object in ``text``, or None."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_score(value: object) -> float | None:
    """Coerce a parsed JSON value to a float in [0, 1], or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            f = float(value.strip())
            return max(0.0, min(1.0, f))
        except ValueError:
            return None
    return None


def _parse_correctness(raw: str) -> CorrectnessResult:
    """Parse one LLM response into a CorrectnessResult, never raising."""
    parsed = _extract_json_object(raw)
    if parsed is None:
        logger.warning("correctness output unparseable output=%r", raw[:200])
        return CorrectnessResult(score=0.0, parse_failed=True)

    score = _parse_score(parsed.get("score"))
    if score is None:
        logger.warning("correctness output missing valid 'score' output=%r", raw[:200])
        return CorrectnessResult(score=0.0, parse_failed=True)

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = None
    return CorrectnessResult(score=score, reason=reason)


def grade_correctness(
    question: str,
    expected_answer: str,
    generated_answer: str,
    *,
    client: llm.LLMClient | None = None,
) -> CorrectnessResult:
    """Judge whether ``generated_answer`` is semantically correct.

    Returns a CorrectnessResult with score in [0, 1]. An empty answer
    or missing expected answer returns score 0.0 without making an LLM
    call. An explicitly supplied client bypasses provider config.
    Provider failures surface as LLMError.
    """

    if not generated_answer or not generated_answer.strip():
        return CorrectnessResult(score=0.0, reason="empty generated answer")
    if not expected_answer or not expected_answer.strip():
        return CorrectnessResult(score=0.0, reason="no expected answer provided")

    llm_client = client if client is not None else llm.get_llm_client()

    prompt_text = build_correctness_prompt(question, expected_answer, generated_answer)
    try:
        raw = llm_client.complete(prompt_text, max_tokens=settings.llm_max_tokens)
    except LLMError:
        raise
    except Exception as exc:
        logger.error("correctness call failed error=%s", exc, exc_info=True)
        raise LLMError(f"Correctness evaluation failed: {exc}") from exc

    return _parse_correctness(raw)
