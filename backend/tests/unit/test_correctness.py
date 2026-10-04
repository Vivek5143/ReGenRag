"""Tests for LLM-based answer correctness evaluation (Phase 5).

All tests use a mocked/fake LLM client. No network, no API keys, no
Gemini/Ollama calls.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import LLMError
from app.evaluation.correctness import (
    CorrectnessResult,
    _parse_correctness,
    _parse_score,
    grade_correctness,
)


class FakeLLMClient:
    """Records calls and returns canned responses or raises."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses) if responses else []
        self.error = error
        self.calls = []

    def complete(self, prompt, *, max_tokens=512):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return ""


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------


def test_parse_score_accepts_float():
    assert _parse_score(0.5) == 0.5
    assert _parse_score(1.0) == 1.0
    assert _parse_score(0.0) == 0.0


def test_parse_score_clamps_below_zero():
    assert _parse_score(-0.5) == 0.0


def test_parse_score_clamps_above_one():
    assert _parse_score(1.5) == 1.0


def test_parse_score_accepts_int():
    assert _parse_score(1) == 1.0
    assert _parse_score(0) == 0.0


def test_parse_score_accepts_numeric_string():
    assert _parse_score("0.75") == 0.75
    assert _parse_score("1") == 1.0


def test_parse_score_rejects_non_numeric_string():
    assert _parse_score("maybe") is None
    assert _parse_score("") is None


def test_parse_score_rejects_bool():
    assert _parse_score(True) is None
    assert _parse_score(False) is None


# ---------------------------------------------------------------------------
# _parse_correctness
# ---------------------------------------------------------------------------


def test_parse_correctness_valid_score():
    result = _parse_correctness('{"score": 1.0, "reason": "correct"}')
    assert result.score == 1.0
    assert result.reason == "correct"
    assert result.parse_failed is False


def test_parse_correctness_malformed_json_sets_parse_failed():
    result = _parse_correctness("not json at all")
    assert result.score == 0.0
    assert result.parse_failed is True
    assert result.reason is None


def test_parse_correctness_missing_score_sets_parse_failed():
    result = _parse_correctness('{"reason": "no score"}')
    assert result.score == 0.0
    assert result.parse_failed is True


def test_parse_correctness_invalid_score_sets_parse_failed():
    result = _parse_correctness('{"score": "maybe"}')
    assert result.score == 0.0
    assert result.parse_failed is True


def test_parse_correctness_missing_reason_is_none():
    result = _parse_correctness('{"score": 0.5}')
    assert result.score == 0.5
    assert result.reason is None
    assert result.parse_failed is False


def test_parse_correctness_empty_reason_becomes_none():
    result = _parse_correctness('{"score": 0.5, "reason": "   "}')
    assert result.reason is None


def test_parse_correctness_clamps_score():
    result = _parse_correctness('{"score": 2.5}')
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# grade_correctness
# ---------------------------------------------------------------------------


def test_grade_correctness_correct_answer():
    fake = FakeLLMClient(responses=['{"score": 1.0, "reason": "matches"}'])
    result = grade_correctness(
        "What is 2+2?",
        "4",
        "The answer is four.",
        client=fake,
    )
    assert isinstance(result, CorrectnessResult)
    assert result.score == 1.0
    assert result.parse_failed is False
    assert len(fake.calls) == 1


def test_grade_correctness_incorrect_answer():
    fake = FakeLLMClient(responses=['{"score": 0.0, "reason": "wrong"}'])
    result = grade_correctness(
        "What is 2+2?",
        "4",
        "The answer is five.",
        client=fake,
    )
    assert result.score == 0.0


def test_grade_correctness_expected_answer_missing():
    """When expected_answer is empty, score is 0.0 and no LLM call is made."""
    fake = FakeLLMClient()
    result = grade_correctness(
        "Q",
        "",
        "A",
        client=fake,
    )
    assert result.score == 0.0
    assert len(fake.calls) == 0


def test_grade_correctness_generated_answer_empty():
    fake = FakeLLMClient()
    result = grade_correctness(
        "Q",
        "A",
        "",
        client=fake,
    )
    assert result.score == 0.0
    assert len(fake.calls) == 0


def test_grade_correctness_semantically_equivalent():
    """Semantic equivalence should score high even with different wording."""
    fake = FakeLLMClient(responses=['{"score": 0.9, "reason": "same meaning"}'])
    result = grade_correctness(
        "What is the capital of France?",
        "Paris",
        "The capital city is Paris.",
        client=fake,
    )
    assert result.score == 0.9


def test_grade_correctness_malformed_judge_output():
    fake = FakeLLMClient(responses=["I think it's correct"])
    result = grade_correctness(
        "Q",
        "A",
        "A",
        client=fake,
    )
    assert result.score == 0.0
    assert result.parse_failed is True


def test_grade_correctness_propagates_llm_error():
    fake = FakeLLMClient(error=LLMError("provider down"))
    with pytest.raises(LLMError):
        grade_correctness("Q", "A", "A", client=fake)


def test_grade_correctness_wraps_unexpected_exception_as_llm_error():
    fake = FakeLLMClient(error=ValueError("boom"))
    with pytest.raises(LLMError):
        grade_correctness("Q", "A", "A", client=fake)


def test_grade_correctness_empty_chunks_makes_no_llm_calls():
    """Empty inputs return a safe result without any LLM interaction."""
    fake = FakeLLMClient()
    result = grade_correctness("", "", "", client=fake)
    assert result.score == 0.0
    assert len(fake.calls) == 0
