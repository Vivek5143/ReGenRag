"""Tests for the session lifecycle service."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.exceptions import SessionNotFoundError
from app.db.enums import SessionStatus
from app.services import session_service


def test_create_session(db_session):
    session = session_service.create_session(db_session)

    assert isinstance(session.id, uuid.UUID)
    assert session.status == SessionStatus.ACTIVE
    assert session.expires_at > session.created_at
    # Rolling expiry should be ~SESSION_TTL_MINUTES ahead.
    assert session.expires_at - session.created_at >= timedelta(minutes=55)


def test_get_session_found(db_session):
    created = session_service.create_session(db_session)

    fetched = session_service.get_session(db_session, created.id)
    assert fetched.id == created.id


def test_get_session_missing_raises(db_session):
    with pytest.raises(SessionNotFoundError):
        session_service.get_session(db_session, uuid.uuid4())


def test_touch_session_extends_expiry(db_session):
    session = session_service.create_session(db_session)
    before_expiry = session.expires_at
    before_activity = session.last_activity

    # Simulate a session that has been idle.
    past = datetime.now(timezone.utc) - timedelta(minutes=30)
    session.last_activity = past
    db_session.commit()

    touched = session_service.touch_session(db_session, session)
    assert touched.last_activity > before_activity
    assert touched.expires_at > before_expiry


def test_close_session(db_session):
    session = session_service.create_session(db_session)

    closed = session_service.close_session(db_session, session.id)
    assert closed.status == SessionStatus.CLOSED


def test_expire_sessions_marks_past_due(db_session):
    session = session_service.create_session(db_session)
    session.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db_session.commit()

    count = session_service.expire_sessions(db_session)
    assert count == 1

    db_session.refresh(session)
    assert session.status == SessionStatus.EXPIRED


def test_expire_sessions_ignores_future_sessions(db_session):
    session = session_service.create_session(db_session)

    count = session_service.expire_sessions(db_session)
    assert count == 0
    db_session.refresh(session)
    assert session.status == SessionStatus.ACTIVE


def test_get_active_session_rejects_expired(db_session):
    session = session_service.create_session(db_session)
    session.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db_session.commit()

    with pytest.raises(Exception):
        session_service.get_active_session(db_session, session.id)