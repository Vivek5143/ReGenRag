"""Domain enums shared by models, services, and API schemas."""

import enum


class SessionStatus(str, enum.Enum):
    """Lifecycle status of a temporary document session."""

    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"


class DocumentStatus(str, enum.Enum):
    """Lifecycle status of an uploaded document."""

    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
    DELETED = "DELETED"