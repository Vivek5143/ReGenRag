"""Temporary document session model.

A Session is the root of the product data model:

    Session
       -> Documents
            -> Chunks
                 -> Embeddings

Sessions are ephemeral: when a session ends or expires, its uploaded PDFs,
chunks, and embeddings are removed. No authentication or users are modelled yet.
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import DateTime, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.db.enums import SessionStatus

if TYPE_CHECKING:
    from app.db.models.document import Document


def utcnow() -> datetime:
    """Timezone-aware UTC "now", used as a Python-side column default."""
    return datetime.now(timezone.utc)


class Session(Base):
    """Represents a temporary, session-scoped document session."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=sa.func.now()
    )
    # Touched on every interaction so expiry reflects real activity.
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=sa.func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[SessionStatus] = mapped_column(
        sa.Enum(SessionStatus, name="session_status"),
        default=SessionStatus.ACTIVE,
        server_default=SessionStatus.ACTIVE.value,
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )