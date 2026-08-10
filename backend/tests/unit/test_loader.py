"""Tests for the PDF loader: extraction, page preservation, and errors."""

import pytest

from app.core.exceptions import DocumentProcessingError
from app.ingestion.loader import load_document
from tests.factories import pdf_bytes


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_extracts_text_from_valid_pdf(tmp_path):
    path = _write(tmp_path, "doc.pdf", pdf_bytes("Hello from the first page."))

    pages = load_document(path, source="doc.pdf")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Hello" in pages[0].text
    assert pages[0].metadata == {"source": "doc.pdf"}


def test_preserves_multiple_pages_and_numbers(tmp_path):
    path = _write(
        tmp_path,
        "multi.pdf",
        pdf_bytes("First page body.", "Second page body.", "Third page body."),
    )

    pages = load_document(path, source="multi.pdf")

    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "First page" in pages[0].text
    assert "Second page" in pages[1].text
    assert "Third page" in pages[2].text


def test_keeps_source_metadata_on_every_page(tmp_path):
    path = _write(tmp_path, "meta.pdf", pdf_bytes("Alpha", "Beta"))

    pages = load_document(path, source="meta.pdf")

    assert all(p.metadata["source"] == "meta.pdf" for p in pages)


def test_rejects_malformed_pdf(tmp_path):
    path = _write(tmp_path, "bad.pdf", b"this is definitely not a pdf")

    with pytest.raises(DocumentProcessingError):
        load_document(path, source="bad.pdf")


def test_rejects_empty_file(tmp_path):
    path = _write(tmp_path, "empty.pdf", b"")

    with pytest.raises(DocumentProcessingError):
        load_document(path, source="empty.pdf")


def test_rejects_zero_page_pdf(tmp_path):
    path = _write(tmp_path, "zero.pdf", pdf_bytes())

    with pytest.raises(DocumentProcessingError):
        load_document(path, source="zero.pdf")


def test_blank_page_is_preserved(tmp_path):
    path = _write(tmp_path, "blank.pdf", pdf_bytes(""))

    pages = load_document(path, source="blank.pdf")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == ""