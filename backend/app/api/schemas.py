"""Pydantic request/response models for the API.

Response models are built explicitly from ORM objects so internal fields
(e.g. physical storage paths) are never leaked to clients.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, field_validator

from app.db.enums import DocumentStatus, SessionStatus


class HealthResponse(BaseModel):
    """Liveness + database connectivity payload."""

    status: str
    service: str
    database: str


class SessionCreateResponse(BaseModel):
    """Payload returned when a new session is created."""

    session_id: uuid.UUID
    status: SessionStatus
    expires_at: datetime


class SessionResponse(BaseModel):
    """Full session metadata."""

    session_id: uuid.UUID
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    expires_at: datetime


class DocumentResponse(BaseModel):
    """Document metadata (never includes the internal storage path).

    ``chunk_count`` reports how many chunks were stored by ingestion
    (0 for an unprocessed or failed document).
    """

    document_id: uuid.UUID
    session_id: uuid.UUID
    filename: str
    status: DocumentStatus
    file_size: int
    chunk_count: int
    created_at: datetime
    processed_at: datetime | None


class QueryRequest(BaseModel):
    """A user question asked against a session's processed documents."""

    question: str

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        """Reject empty/whitespace-only questions and normalize the input."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class QuerySource(BaseModel):
    """A single retrieved chunk referenced as a source for an answer.

    ``score`` is the cosine similarity of the chunk to the query embedding
    (higher is more relevant).
    """

    document_id: uuid.UUID
    chunk_id: uuid.UUID
    chunk_index: int
    page_number: int | None
    score: float | None


class QueryResponse(BaseModel):
    """Result of a RAG query: the generated answer plus its sources."""

    answer: str
    sources: list[QuerySource]