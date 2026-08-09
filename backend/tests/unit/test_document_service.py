"""Tests for the document upload/storage service."""

import io

import pytest
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import DocumentValidationError
from app.db.enums import DocumentStatus, SessionStatus
from app.services import document_service, session_service

PDF_BYTES = b"%PDF-1.4 fake pdf content for testing"


def _upload(name="paper.pdf", content=None, content_type="application/pdf"):
    return UploadFile(
        filename=name,
        file=io.BytesIO(content or PDF_BYTES),
        headers={"content-type": content_type},
    )


def test_upload_valid_pdf(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)

    doc = document_service.upload_document(db_session, session.id, _upload())

    assert doc.status == DocumentStatus.UPLOADED
    assert doc.session_id == session.id
    assert doc.file_size == len(PDF_BYTES)
    assert doc.filename == "paper.pdf"
    # File exists on disk under a safe generated path.
    assert (tmp_path / doc.storage_path).exists()
    assert doc.stored_filename.endswith(".pdf")


def test_reject_non_pdf_extension(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)

    with pytest.raises(DocumentValidationError):
        document_service.upload_document(
            db_session, session.id, _upload(name="notes.txt", content_type="text/plain")
        )


def test_reject_wrong_content_type(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)

    with pytest.raises(DocumentValidationError):
        document_service.upload_document(
            db_session, session.id, _upload(name="paper.pdf", content_type="text/html")
        )


def test_reject_oversized_file(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    monkeypatch.setattr(settings, "max_file_size_mb", 0.001)  # ~1 KB
    session = session_service.create_session(db_session)

    with pytest.raises(DocumentValidationError):
        document_service.upload_document(
            db_session, session.id, _upload(name="big.pdf", content=b"x" * 4096)
        )


def test_upload_to_expired_session_rejected(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)
    session.status = SessionStatus.EXPIRED
    db_session.commit()

    with pytest.raises(Exception):
        document_service.upload_document(db_session, session.id, _upload())


def test_document_belongs_to_correct_session(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session_a = session_service.create_session(db_session)
    session_b = session_service.create_session(db_session)

    doc = document_service.upload_document(db_session, session_a.id, _upload())
    assert doc.session_id == session_a.id
    assert doc.session_id != session_b.id


def test_delete_document_removes_record_and_file(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)
    doc = document_service.upload_document(db_session, session.id, _upload())
    stored = tmp_path / doc.storage_path
    assert stored.exists()

    document_service.delete_document(db_session, doc.id)

    assert not stored.exists()