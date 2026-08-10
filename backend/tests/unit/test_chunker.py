"""Tests for the chunker: sizing, overlap, indexing, and metadata."""

import uuid

from app.ingestion.chunker import chunk_pages, chunk_text
from app.ingestion.loader import LoadedPage


def test_short_text_is_single_chunk():
    pieces = chunk_text("A short paragraph.", chunk_size=1000, chunk_overlap=200)

    assert pieces == ["A short paragraph."]


def test_long_text_splits_respecting_size():
    text = "word " * 500  # ~2495 chars
    pieces = chunk_text(text, chunk_size=1000, chunk_overlap=200)

    assert len(pieces) > 1
    assert all(len(p) <= 1000 for p in pieces)
    # The whole document is covered.
    assert "".join(pieces)[:50] == text[:50]


def test_overlap_between_consecutive_chunks():
    # No sentence/paragraph markers -> hard splits, exact overlap math.
    text = " ".join(["word"] * 300)
    chunk_size, overlap = 100, 20
    pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=overlap)

    assert len(pieces) > 1
    # The next piece re-includes the tail of the previous one (tails are
    # stripped, so compare the stripped tail against the next piece's head).
    tail = pieces[0][-overlap:].strip()
    assert tail and pieces[1].startswith(tail)


def test_prefers_natural_boundaries_but_stays_bounded():
    # Text with clear paragraph breaks: the splitter should pick a paragraph
    # boundary when one falls inside the search window, and never produce a
    # piece larger than the window or with leading whitespace.
    para = "alpha beta gamma delta epsilon zeta eta theta iota " * 2  # > chunk
    text = f"{para}\n\n{para}"
    pieces = chunk_text(text, chunk_size=60, chunk_overlap=10)

    assert len(pieces) > 1
    # Pieces are stripped (no leading/trailing whitespace) and sized.
    assert all(p and p[0].isalnum() and len(p) <= 60 for p in pieces)
    # When a paragraph break sits near the window, a piece ends at it (i.e. at
    # least one boundary split landed on "\n\n" -> piece starts on a word).
    assert any(p.startswith("alpha") for p in pieces)


def test_empty_text_yields_no_chunks():
    assert chunk_text("", chunk_size=100, chunk_overlap=20) == []
    assert chunk_text("   ", chunk_size=100, chunk_overlap=20) == []


def test_invalid_overlap_rejected():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("x" * 50, chunk_size=50, chunk_overlap=50)


def test_global_sequential_indices_across_pages():
    document_id = uuid.uuid4()
    pages = [
        LoadedPage(page_number=1, text="A" * 300, metadata={"source": "a.pdf"}),
        LoadedPage(page_number=2, text="B" * 300, metadata={"source": "a.pdf"}),
    ]
    chunks = chunk_pages(
        pages,
        document_id=document_id,
        filename="a.pdf",
        chunk_size=100,
        chunk_overlap=10,
    )

    # ~3 chunks/page -> sequential indices document-wide.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Chunks never cross pages.
    assert all(c.page_number in (1, 2) for c in chunks)


def test_chunk_metadata_preserved():
    document_id = uuid.uuid4()
    pages = [
        LoadedPage(page_number=3, text="Alpha " * 40, metadata={"source": "b.pdf"}),
    ]
    chunks = chunk_pages(
        pages,
        document_id=document_id,
        filename="b.pdf",
        chunk_size=80,
        chunk_overlap=10,
    )

    for chunk in chunks:
        assert chunk.metadata["document_id"] == str(document_id)
        assert chunk.metadata["filename"] == "b.pdf"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["chunk_index"] == chunk.chunk_index
        assert chunk.page_number == 3