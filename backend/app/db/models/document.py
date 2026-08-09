"""Uploaded document model.

Documents belong to exactly one session and derive their lifetime from it.
Deleting a session cascades to its documents (and, through documents, to their
chunks). This phase only models the upload/storage lifecycle: no text
extraction, chunking, or embeddings are performed yet.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import DateTime, ForeignKey, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.db.enums import DocumentStatus
from app.db.models.session import utcnow

if TYPE_CHECKING:
    from app.db.models.document_chunk import DocumentChunk
    from app.db.models.session import Session


class Document(Base):
    """Represents one uploaded PDF within a session."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    # Original client filename (safe, display-only).
    filename: Mapped[str] = mapped_column(String(255))
    # Server-generated filename; never trusts the client for storage.
    stored_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer)
    # Relative path under the configured upload dir, e.g. "<session>/<doc>.pdf".
    storage_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[DocumentStatus] = mapped_column(
        sa.Enum(DocumentStatus, name="document_status"),
        default=DocumentStatus.UPLOADED,
        server_default=DocumentStatus.UPLOADED.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=sa.func.now()
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    session: Mapped["Session"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )