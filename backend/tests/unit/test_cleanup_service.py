"""Tests for session/document cleanup, including idempotency."""

import io
from datetime import datetime, timedelta, timezone

from fastapi import UploadFile

from app.core.config import settings
from app.db.enums import SessionStatus
from app.db.models.document import Document
from app.db.models.session import Session
from app.services import cleanup_service, document_service, session_service


def _upload(name="doc.pdf", content=b"%PDF-1.4 fake"):
    return UploadFile(
        filename=name,
        file=io.BytesIO(content),
        headers={"content-type": "application/pdf"},
    )


def test_cleanup_expired_removes_records_and_files(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)
    doc = document_service.upload_document(db_session, session.id, _upload())
    stored = tmp_path / doc.storage_path
    assert stored.exists()

    session.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db_session.commit()

    cleaned = cleanup_service.cleanup_expired_sessions(
        db_session, upload_dir=tmp_path
    )

    assert cleaned == 1
    assert db_session.get(type(doc), doc.id) is None  # document record removed
    assert not stored.exists()  # physical file removed


def test_cleanup_is_idempotent(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)
    document_service.upload_document(db_session, session.id, _upload())

    session.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db_session.commit()

    first = cleanup_service.cleanup_expired_sessions(db_session, upload_dir=tmp_path)
    second = cleanup_service.cleanup_expired_sessions(db_session, upload_dir=tmp_path)

    assert first == 1
    assert second == 0


def test_close_then_cleanup_session_removes_everything(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    session = session_service.create_session(db_session)
    doc = document_service.upload_document(db_session, session.id, _upload())
    stored = tmp_path / doc.storage_path

    session_service.close_session(db_session, session.id)
    count = cleanup_service.cleanup_session(db_session, session.id, upload_dir=tmp_path)

    assert count == 1
    assert not stored.exists()
    assert db_session.get(Session, session.id).status == SessionStatus.CLOSED