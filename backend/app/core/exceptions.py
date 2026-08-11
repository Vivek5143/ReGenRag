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


class DocumentProcessingError(ReGenRAGError):
    """PDF parsing, chunking, embedding, or persistence failed during ingestion."""


class NoProcessedDocumentError(ReGenRAGError):
    """A session has no processed document to query against."""


class RetrievalError(ReGenRAGError):
    """Vector similarity retrieval failed (embedding, database, or search)."""


class LLMConfigError(ReGenRAGError):
    """No LLM provider is configured, or its configuration is invalid."""


class LLMError(ReGenRAGError):
    """An LLM provider call failed (network, HTTP, or malformed response)."""