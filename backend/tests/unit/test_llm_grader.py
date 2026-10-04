"""Unit tests for LLM-based retrieval relevance grading (Phase 4, step 2).

These tests mock the LLM so no real provider calls are made.
"""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.llm_grader import (
    LlmRetrievalGrading,
    _coerce_bool,
    _extract_json_object,
    _parse_judgement,
    grade_relevance,
)
from app.retrieval.vector_store import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content="evidence text",
        score=0.9,
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
# _coerce_bool
# ---------------------------------------------------------------------------


def test_coerce_bool_accepts_true_strings():
    assert _coerce_bool("true") is True
    assert _coerce_bool("True") is True
    assert _coerce_bool("yes") is True
    assert _coerce_bool("1") is True


def test_coerce_bool_accepts_false_strings():
    assert _coerce_bool("false") is False
    assert _coerce_bool("no") is False
    assert _coerce_bool("0") is False


def test_coerce_bool_passes_through_native_bools():
    assert _coerce_bool(True) is True
    assert _coerce_bool(False) is False


def test_coerce_bool_returns_none_for_ambiguous_values():
    assert _coerce_bool("maybe") is None
    assert _coerce_bool("") is None
    assert _coerce_bool(None) is None
    assert _coerce_bool(42) is None


# ---------------------------------------------------------------------------
# _extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_object_finds_balanced_object():
    text = 'Here is the result: {"relevant": true, "reason": "ok"} done'
    parsed = _extract_json_object(text)
    assert parsed == {"relevant": True, "reason": "ok"}


def test_extract_json_object_strips_markdown_code_fences():
    text = '```json\n{"relevant": false}\n```'
    parsed = _extract_json_object(text)
    assert parsed == {"relevant": False}


def test_extract_json_object_returns_none_when_no_braces():
    assert _extract_json_object("no json here") is None


def test_extract_json_object_returns_none_for_unbalanced():
    assert _extract_json_object('{"relevant": true') is None


def test_extract_json_object_returns_none_for_non_dict():
    assert _extract_json_object("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# _parse_judgement
# ---------------------------------------------------------------------------


def test_parse_judgement_valid_true():
    chunk_id = uuid.uuid4()
    judgement = _parse_judgement(chunk_id, '{"relevant": true, "reason": "matches"}')
    assert judgement.chunk_id == chunk_id
    assert judgement.relevant is True
    assert judgement.reason == "matches"
    assert judgement.parse_failed is False


def test_parse_judgement_valid_false():
    chunk_id = uuid.uuid4()
    judgement = _parse_judgement(chunk_id, '{"relevant": false, "reason": "irrelevant"}')
    assert judgement.relevant is False
    assert judgement.reason == "irrelevant"
    assert judgement.parse_failed is False


def test_parse_judgement_malformed_json_sets_parse_failed():
    chunk_id = uuid.uuid4()
    judgement = _parse_judgement(chunk_id, "not json at all")
    assert judgement.relevant is False
    assert judgement.parse_failed is True
    assert judgement.reason is None


def test_parse_judgement_missing_relevant_field_sets_parse_failed():
    chunk_id = uuid.uuid4()
    judgement = _parse_judgement(chunk_id, '{"reason": "no relevant field"}')
    assert judgement.relevant is False
    assert judgement.parse_failed is True


def test_parse_judgement_ambiguous_relevant_value_sets_parse_failed():
    chunk_id = uuid.uuid4()
    judgement = _parse_judgement(chunk_id, '{"relevant": "maybe"}')
    assert judgement.relevant is False
    assert judgement.parse_failed is True


def test_parse_judgement_missing_reason_is_none():
    chunk_id = uuid.uuid4()
    judgement = _parse_judgement(chunk_id, '{"relevant": true}')
    assert judgement.reason is None
    assert judgement.parse_failed is False


def test_parse_judgement_empty_reason_becomes_none():
    chunk_id = uuid.uuid4()
    judgement = _parse_judgement(chunk_id, '{"relevant": true, "reason": "   "}')
    assert judgement.reason is None
    assert judgement.parse_failed is False


# ---------------------------------------------------------------------------
# grade_relevance
# ---------------------------------------------------------------------------


def test_grade_relevance_judges_each_chunk():
    chunk = _chunk()
    fake = FakeLLMClient(responses=['{"relevant": true, "reason": "ok"}'])
    grading = grade_relevance("question", [chunk], client=fake)
    assert len(grading.judgements) == 1
    assert grading.judgements[0].relevant is True
    assert grading.relevant_count == 1
    assert grading.has_relevant_evidence is True
    assert grading.failed_count == 0
    assert len(fake.calls) == 1


def test_grade_relevance_empty_chunks_makes_no_llm_calls():
    grading = grade_relevance("question", [], client=FakeLLMClient())
    assert grading.judgements == []
    assert grading.relevant_count == 0
    assert grading.has_relevant_evidence is False
    assert grading.reason == "no chunks to grade"


def test_grade_relevance_aggregates_mixed_verdicts():
    chunks = [_chunk(), _chunk()]
    fake = FakeLLMClient(
        responses=[
            '{"relevant": true, "reason": "a"}',
            '{"relevant": false, "reason": "b"}',
        ]
    )
    grading = grade_relevance("question", chunks, client=fake)
    assert grading.relevant_count == 1
    assert grading.has_relevant_evidence is True
    assert grading.failed_count == 0


def test_grade_relevance_propagates_llm_error():
    from app.core.exceptions import LLMError

    chunk = _chunk()
    fake = FakeLLMClient(error=LLMError("provider down"))
    with pytest.raises(LLMError):
        grade_relevance("question", [chunk], client=fake)


def test_grade_relevance_wraps_unexpected_exception_as_llm_error():
    from app.core.exceptions import LLMError

    chunk = _chunk()
    fake = FakeLLMClient(error=ValueError("boom"))
    with pytest.raises(LLMError):
        grade_relevance("question", [chunk], client=fake)


def test_grade_relevance_counts_parse_failures():
    chunks = [_chunk(), _chunk()]
    fake = FakeLLMClient(
        responses=[
            '{"relevant": true, "reason": "ok"}',
            "not json",
        ]
    )
    grading = grade_relevance("question", chunks, client=fake)
    assert grading.relevant_count == 1
    assert grading.failed_count == 1
    assert grading.has_relevant_evidence is True