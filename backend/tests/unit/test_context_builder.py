"""Tests for the context builder: retrieved chunks -> LLM-readable context."""

import uuid

from app.retrieval import context_builder
from app.retrieval.vector_store import RetrievedChunk


def _chunk(page, text, index=0):
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=index,
        page_number=page,
        content=text,
        score=0.5,
    )


def test_page_numbers_preserved():
    chunks = [_chunk(4, "text four"), _chunk(7, "text seven"), _chunk(14, "text fourteen")]

    context = context_builder.build_context(chunks)

    assert "[Source: Page 4]" in context
    assert "[Source: Page 7]" in context
    assert "[Source: Page 14]" in context


def test_multiple_chunks_formatted_in_order():
    chunks = [_chunk(4, "alpha"), _chunk(7, "beta")]

    context = context_builder.build_context(chunks)

    assert context == "[Source: Page 4]\nalpha\n\n[Source: Page 7]\nbeta"


def test_chunk_content_kept_verbatim():
    chunk = _chunk(2, "The quick brown fox jumps over the lazy dog.")

    context = context_builder.build_context([chunk])

    assert "The quick brown fox jumps over the lazy dog." in context


def test_empty_chunks():
    assert context_builder.build_context([]) == ""


def test_missing_page_number_renders_unknown():
    chunk = _chunk(None, "no page")

    context = context_builder.build_context([chunk])

    assert context == "[Source: Page unknown]\nno page"
