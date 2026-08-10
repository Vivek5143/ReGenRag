"""Tests for the ingestion service orchestration and status lifecycle.

Uses a fake embedding model (no downloads) so the pipeline runs end-to-end
against the test database with a real, reportlab-generated PDF.
"""

import io

import pytest
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.services import document_service, ingestion_service, session_service
from tests.factories import pdf_bytes


def _upload(name, content):
    return UploadFile(
        filename=name, file=io.BytesIO(content), headers={"content-type": "application/pdf"}
    )


def _uploaded_document(db_session, tmp_path, monkeypatch, name="report.pdf", content=None):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)
    content = content if content is not None else pdf_bytes("The quick brown fox." * 30)
    return document_service.upload_document(db_session, session.id, _upload(name, content))


def _chunk_count(db_session, document_id):
    return (
        db_session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .count()
    )


def test_successful_ingestion_marks_processed(db_session, tmp_path, monkeypatch):
    document = _uploaded_document(db_session, tmp_path, monkeypatch)
    assert document.status == DocumentStatus.UPLOADED

    processed = ingestion_service.ingest_document(db_session, document)

    assert processed.status == DocumentStatus.PROCESSED
    assert processed.processed_at is not None


def test_chunks_and_embeddings_persisted(db_session, tmp_path, monkeypatch):
    document = _uploaded_document(db_session, tmp_path, monkeypatch)

    ingestion_service.ingest_document(db_session, document)

    chunks = (
        db_session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    assert len(chunks) >= 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384
        assert chunk.metadata["document_id"] == str(document.id)
        assert chunk.metadata["filename"] == document.filename


def test_failure_marks_document_failed(db_session, tmp_path, monkeypatch):
    # Fake bytes -> pypdf cannot parse -> ingestion fails.
    document = _uploaded_document(
        db_session, tmp_path, monkeypatch, content=b"%PDF-1.4 not really a pdf"
    )

    with pytest.raises(DocumentProcessingError):
        ingestion_service.ingest_document(db_session, document)

    db_session.expire_all()
    stored = db_session.get(Document, document.id)
    assert stored.status == DocumentStatus.FAILED
    assert _chunk_count(db_session, document.id) == 0


def test_failure_rolls_back_partial_chunks(db_session, tmp_path, monkeypatch):
    document = _uploaded_document(db_session, tmp_path, monkeypatch)

    def broken_persist(db, doc, chunks, vectors):
        # Simulate partial DB writes before a failure mid-persist.
        db.add(
            DocumentChunk(
                document_id=doc.id, chunk_index=0,
                content="partial", embedding=[0.0] * 384,
            )
        )
        db.flush()
        raise RuntimeError("db exploded mid-persist")

    monkeypatch.setattr(ingestion_service, "_persist_chunks", broken_persist)

    with pytest.raises(DocumentProcessingError):
        ingestion_service.ingest_document(db_session, document)

    db_session.expire_all()
    # The partial chunk was rolled back; the document is FAILED, not stuck.
    assert _chunk_count(db_session, document.id) == 0
    assert db_session.get(Document, document.id).status == DocumentStatus.FAILED


def test_document_not_left_in_processing_on_failure(db_session, tmp_path, monkeypatch):
    document = _uploaded_document(
        db_session, tmp_path, monkeypatch, content=b"garbage"
    )

    with pytest.raises(DocumentProcessingError):
        ingestion_service.ingest_document(db_session, document)

    db_session.expire_all()
    assert db_session.get(Document, document.id).status == DocumentStatus.FAILED