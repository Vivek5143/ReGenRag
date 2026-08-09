"""Document chunk model.

Establishes the schema for extracted document chunks. Content, page metadata,
and an empty embedding column are prepared here, but the actual chunking and
embedding logic are intentionally not implemented until a later phase.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.database import Base
from app.db.models.session import utcnow

if TYPE_CHECKING:
    from app.db.models.document import Document


class DocumentChunk(Base):
    """A single chunk of a processed document."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # JSONB holds per-chunk metadata (e.g. source, headings) once ingestion
    # exists. The attribute is `meta` because `metadata` is reserved in the
    # SQLAlchemy Declarative API; the database column stays "metadata".
    meta: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    # pgvector column, prepared but intentionally unused in Phase 1. Its
    # dimension is configurable (EMBEDDING_DIMENSION) and must match the
    # embedding model selected later. No embeddings are generated or stored now.
    embedding: Mapped[list | None] = mapped_column(
        Vector(settings.embedding_dimension), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=sa.func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")