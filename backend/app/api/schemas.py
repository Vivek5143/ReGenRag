"""Pydantic request/response models for the API.

Response models are built explicitly from ORM objects so internal fields
(e.g. physical storage paths) are never leaked to clients.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel

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
    """Document metadata (never includes the internal storage path)."""

    document_id: uuid.UUID
    session_id: uuid.UUID
    filename: str
    status: DocumentStatus
    file_size: int
    created_at: datetime
    processed_at: datetime | None