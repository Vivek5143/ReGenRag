"""Unit tests for Phase 4 query rewriting.

These tests mock the LLM so no real provider calls are made.
"""

from __future__ import annotations

import uuid

import pytest

from app.core.exceptions import LLMError
from app.retrieval import llm_grader as llm_grader_module
from app.retrieval import similarity_grader
from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.similarity_grader import RetrievalGrading
from app.retrieval.vector_store import RetrievedChunk


DIM = 1


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content="irrelevant but decoy chunk",
        score=score,
    )


class FakeLLMClient:
    def __init__(self, *, response: str | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        return self.response or ""


def _llm_relevance(*, has_relevant_evidence: bool) -> llm_grader_module.LlmRetrievalGrading:
    return llm_grader_module.LlmRetrievalGrading(
        judgements=[],
        relevant_count=1 if has_relevant_evidence else 0,
        failed_count=0,
        has_relevant_evidence=has_relevant_evidence,
        reason="ok" if has_relevant_evidence else "no relevant evidence",
    )


def test_query_rewrite_calls_llm_and_returns_non_empty_string():
    """Test 1 — query rewriter itself."""

    chunks = [_chunk(0.1), _chunk(0.2)]
    retrieval_grading: RetrievalGrading = similarity_grader.grade_retrieval(
        chunks, threshold=0.65
    )
    assert retrieval_grading.sufficient is False

    expected = "Find sections about quarterly revenue performance and outlook"

    fake_llm = FakeLLMClient(response=expected)

    rewritten = rewrite_query(
        "What does the report say?",
        chunks=chunks,
        retrieval_grading=retrieval_grading,
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
        client=fake_llm,
    )

    assert rewritten == expected
    assert rewritten.strip()
    assert len(fake_llm.calls) == 1


def test_query_rewrite_returns_none_on_empty_output():
    chunks = [_chunk(0.1)]
    retrieval_grading = similarity_grader.grade_retrieval(chunks, threshold=0.65)

    fake_llm = FakeLLMClient(response="   ")

    rewritten = rewrite_query(
        "What does the report say?",
        chunks=chunks,
        retrieval_grading=retrieval_grading,
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
        client=fake_llm,
    )

    assert rewritten is None


def test_query_rewrite_returns_none_on_llm_error():
    chunks = [_chunk(0.1)]
    retrieval_grading = similarity_grader.grade_retrieval(chunks, threshold=0.65)

    fake_llm = FakeLLMClient(error=LLMError("provider down"))

    rewritten = rewrite_query(
        "What does the report say?",
        chunks=chunks,
        retrieval_grading=retrieval_grading,
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
        client=fake_llm,
    )

    assert rewritten is None


def test_query_rewrite_handles_json_in_markdown_code_fences():
    """Test that JSON wrapped in ```json ... ``` is properly extracted."""
    chunks = [_chunk(0.1)]
    retrieval_grading = similarity_grader.grade_retrieval(chunks, threshold=0.65)

    # LLM returns JSON wrapped in markdown code fences (common behavior)
    json_with_fences = """```json
{"rewritten_query": "Find sections about quarterly revenue performance and outlook"}
```"""

    fake_llm = FakeLLMClient(response=json_with_fences)

    rewritten = rewrite_query(
        "What does the report say?",
        chunks=chunks,
        retrieval_grading=retrieval_grading,
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
        client=fake_llm,
    )

    assert rewritten == "Find sections about quarterly revenue performance and outlook"
    assert not rewritten.startswith("```")


def test_query_rewrite_handles_json_without_code_fences():
    """Test that bare JSON still works."""
    chunks = [_chunk(0.1)]
    retrieval_grading = similarity_grader.grade_retrieval(chunks, threshold=0.65)

    bare_json = '{"rewritten_query": "Find sections about quarterly revenue performance and outlook"}'

    fake_llm = FakeLLMClient(response=bare_json)

    rewritten = rewrite_query(
        "What does the report say?",
        chunks=chunks,
        retrieval_grading=retrieval_grading,
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
        client=fake_llm,
    )

    assert rewritten == "Find sections about quarterly revenue performance and outlook"


def test_query_rewrite_handles_plain_text_response():
    """Test that plain text responses (non-JSON) are returned as-is."""
    chunks = [_chunk(0.1)]
    retrieval_grading = similarity_grader.grade_retrieval(chunks, threshold=0.65)

    plain_text = "What are the key financial metrics?"

    fake_llm = FakeLLMClient(response=plain_text)

    rewritten = rewrite_query(
        "What does the report say?",
        chunks=chunks,
        retrieval_grading=retrieval_grading,
        llm_relevance=_llm_relevance(has_relevant_evidence=False),
        client=fake_llm,
    )

    assert rewritten == plain_text
