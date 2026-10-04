"""Tests for the evaluation data model.

Verifies :class:`EvalCase` validation: required fields, type coercion, and
rejection of malformed inputs — all without database or network.
"""

from __future__ import annotations

import uuid

import pytest

from app.evaluation.model import EvalCase


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_valid_case_minimal():
    case = EvalCase(id="q1", query="What is the capital?")
    assert case.id == "q1"
    assert case.query == "What is the capital?"
    assert case.expected_answer is None
    assert case.relevant_chunk_ids is None
    assert case.relevant_document_ids is None
    assert case.metadata is None


def test_valid_case_full():
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    case = EvalCase(
        id="q2",
        query="Where was Newton born?",
        expected_answer="Woolsthorpe Manor",
        relevant_document_ids=[doc_id],
        relevant_chunk_ids=[chunk_id],
        metadata={"difficulty": "easy"},
    )
    assert case.relevant_chunk_ids == [chunk_id]
    assert case.relevant_document_ids == [doc_id]
    assert case.metadata == {"difficulty": "easy"}


# ---------------------------------------------------------------------------
# ID validation
# ---------------------------------------------------------------------------


def test_blank_id_rejected():
    with pytest.raises(ValueError, match="id must not be blank"):
        EvalCase(id="", query="q")


def test_whitespace_only_id_rejected():
    with pytest.raises(ValueError, match="id must not be blank"):
        EvalCase(id="   ", query="q")


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------


def test_blank_query_rejected():
    with pytest.raises(ValueError, match="query must not be blank"):
        EvalCase(id="q1", query="")


def test_whitespace_only_query_rejected():
    with pytest.raises(ValueError, match="query must not be blank"):
        EvalCase(id="q1", query="   ")


# ---------------------------------------------------------------------------
# UUID coercion from strings
# ---------------------------------------------------------------------------


def test_chunk_ids_coerced_from_strings():
    sid = str(uuid.uuid4())
    case = EvalCase(id="q1", query="q", relevant_chunk_ids=[sid])
    assert len(case.relevant_chunk_ids) == 1
    assert case.relevant_chunk_ids[0] == uuid.UUID(sid)


def test_doc_ids_coerced_from_strings():
    sid = str(uuid.uuid4())
    case = EvalCase(id="q1", query="q", relevant_document_ids=[sid])
    assert case.relevant_document_ids[0] == uuid.UUID(sid)


def test_invalid_uuid_string_raises():
    with pytest.raises(ValueError):
        EvalCase(id="q1", query="q", relevant_chunk_ids=["not-a-uuid"])


def test_non_list_chunk_ids_raises():
    with pytest.raises(ValueError, match="relevant_chunk_ids must be a list"):
        EvalCase(id="q1", query="q", relevant_chunk_ids="single-id")


def test_non_list_doc_ids_raises():
    with pytest.raises(ValueError, match="relevant_document_ids must be a list"):
        EvalCase(id="q1", query="q", relevant_document_ids="single-id")


# ---------------------------------------------------------------------------
# None handling
# ---------------------------------------------------------------------------


def test_none_oracle_is_allowed():
    """An eval case without ground truth is valid — metrics will just return 0."""
    case = EvalCase(id="q1", query="q")
    assert case.relevant_chunk_ids is None
    assert case.relevant_document_ids is None
