"""Domain exceptions raised by services and mapped to HTTP statuses in routes."""


class ReGenRAGError(Exception):
    """Base class for application domain errors."""


class SessionNotFoundError(ReGenRAGError):
    """A requested session does not exist."""


class SessionNotActiveError(ReGenRAGError):
    """A session is not active (expired/closed) and cannot receive documents."""


class DocumentNotFoundError(ReGenRAGError):
    """A requested document does not exist."""


class DocumentValidationError(ReGenRAGError):
    """An upload failed validation (type, size, path)."""