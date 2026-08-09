"""Session lifecycle service.

A Session is the root of the product data model and everything related to
uploaded documents belongs to a session. This service owns the lifecycle:

    create -> (activity) -> close / expire -> cleanup

Rolling expiry: every interaction (touch) extends ``expires_at`` by
SESSION_TTL_MINUTES.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import update
from sqlalchemy.orm import Session as DbSession

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError, SessionNotActiveError
from app.core.logging import get_logger
from app.db.enums import SessionStatus
from app.db.models.session import Session

logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expiry_from(now: datetime) -> datetime:
    return now + timedelta(minutes=settings.session_ttl_minutes)


def create_session(db: DbSession) -> Session:
    """Create a new active session with a UUID id and a rolling expiry."""
    now = _now()
    session = Session(
        created_at=now,
        last_activity=now,
        expires_at=_expiry_from(now),
        status=SessionStatus.ACTIVE,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info(
        "session created session_id=%s status=%s expires_at=%s",
        session.id, session.status.value, session.expires_at,
    )
    return session


def get_session(db: DbSession, session_id) -> Session:
    """Return a session by id, raising if it does not exist."""
    session = db.get(Session, session_id)
    if session is None:
        raise SessionNotFoundError(f"Session {session_id} not found")
    return session


def get_active_session(db: DbSession, session_id) -> Session:
    """Return a session that may still receive documents, else raise."""
    session = get_session(db, session_id)
    if session.status != SessionStatus.ACTIVE or session.expires_at <= _now():
        raise SessionNotActiveError(
            f"Session {session_id} is not active (status={session.status.value})"
        )
    return session


def touch_session(db: DbSession, session: Session) -> Session:
    """Refresh a session's activity and extend its rolling expiry."""
    now = _now()
    session.last_activity = now
    session.expires_at = _expiry_from(now)
    db.commit()
    db.refresh(session)
    return session


def close_session(db: DbSession, session_id) -> Session:
    """Mark a session CLOSED. Callers trigger content cleanup separately."""
    session = get_session(db, session_id)
    session.status = SessionStatus.CLOSED
    db.commit()
    db.refresh(session)
    logger.info("session closed session_id=%s", session.id)
    return session


def expire_sessions(db: DbSession, now: datetime | None = None) -> int:
    """Mark ACTIVE sessions past their expiry as EXPIRED. Returns the count."""
    now = now or _now()
    result = db.execute(
        update(Session)
        .where(Session.status == SessionStatus.ACTIVE, Session.expires_at <= now)
        .values(status=SessionStatus.EXPIRED)
    )
    db.commit()
    count = result.rowcount or 0
    if count:
        logger.info("sessions expired count=%s", count)
    return count