"""Cleanup of expired/closed sessions.

Deletes a session's documents (cascading to chunks via the database) and the
associated PDF files on disk. Idempotent: running cleanup a second time finds
nothing to do. No scheduler is built yet; this is the service a scheduler (or
an explicit session-close) calls later.
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session as DbSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.enums import SessionStatus
from app.db.models.document import Document
from app.db.models.session import Session
from app.services import storage
from app.services.session_service import expire_sessions

logger = get_logger(__name__)


def _cleanup_session_contents(
    db: DbSession, session: Session, upload_dir: Path
) -> int:
    """Delete a session's documents (cascades chunks) and their files."""
    documents = db.scalars(
        select(Document).where(Document.session_id == session.id)
    ).all()
    for doc in documents:
        storage.delete_uploaded_file(doc.storage_path, upload_dir)
    if documents:
        db.execute(delete(Document).where(Document.session_id == session.id))
    return len(documents)


def cleanup_expired_sessions(
    db: DbSession, now: datetime | None = None, upload_dir: Path | None = None
) -> int:
    """Expire past-due sessions and remove their documents, files, and chunks.

    Returns the number of documents cleaned up. Idempotent.
    """
    upload_root = Path(upload_dir or settings.upload_dir)
    expire_sessions(db, now)

    sessions = db.scalars(
        select(Session).where(
            Session.status.in_([SessionStatus.EXPIRED, SessionStatus.CLOSED])
        )
    ).all()
    total = 0
    for session in sessions:
        total += _cleanup_session_contents(db, session, upload_root)
    db.commit()
    if total:
        logger.info(
            "cleanup executed sessions=%s documents=%s", len(sessions), total
        )
    return total


def cleanup_session(
    db: DbSession, session_id, upload_dir: Path | None = None
) -> int:
    """Remove a single session's contents (files + records), used on close."""
    upload_root = Path(upload_dir or settings.upload_dir)
    session: Session | None = db.get(Session, session_id)
    if session is None:
        return 0
    count = _cleanup_session_contents(db, session, upload_root)
    db.commit()
    return count