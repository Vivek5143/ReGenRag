"""Document upload/storage lifecycle.

Establishes the upload -> database record -> (eventual processing) lifecycle.
PDF parsing, chunking, and embeddings are intentionally NOT implemented here.
"""

import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session as DbSession

from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError, DocumentValidationError
from app.core.logging import get_logger
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.services import session_service, storage

logger = get_logger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",  # some clients send this for PDFs
}
_CHUNK_SIZE = 64 * 1024


def validate_upload(filename: str | None, content_type: str | None) -> None:
    """Validate upload type. Size is enforced while streaming the file."""
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise DocumentValidationError("Only PDF files are supported.")
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise DocumentValidationError(f"Unsupported content type: {content_type}")


def save_upload_file(
    file: UploadFile,
    session_id,
    document_id,
    max_bytes: int,
    upload_dir: Path | None = None,
) -> tuple[str, int]:
    """Stream the upload to ``<upload_dir>/<session_id>/<document_id>.pdf``.

    Enforces the size limit while streaming and removes the partial file on
    violation. Returns ``(relative_path, bytes_written)``.
    """
    upload_root = Path(upload_dir or settings.upload_dir)
    session_dir = upload_root / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / f"{document_id}.pdf"
    size = 0
    oversized = False
    with dest.open("wb") as out:
        while True:
            chunk = file.file.read(_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                oversized = True
                break
            out.write(chunk)
    # Unlink only after the write handle is closed (required on Windows).
    if oversized:
        dest.unlink(missing_ok=True)
        raise DocumentValidationError(
            "File exceeds the maximum allowed size."
        )
    return f"{session_id}/{document_id}.pdf", size


def upload_document(db: DbSession, session_id, file: UploadFile) -> Document:
    """Validate, store, and register an uploaded PDF for a session."""
    session = session_service.get_active_session(db, session_id)
    validate_upload(file.filename, file.content_type)

    document_id = uuid.uuid4()
    relative_path, size = save_upload_file(
        file, session_id, document_id, settings.max_upload_bytes
    )

    document = Document(
        id=document_id,
        session_id=session_id,
        filename=file.filename or f"{document_id}.pdf",
        stored_filename=f"{document_id}.pdf",
        content_type=file.content_type,
        file_size=size,
        storage_path=relative_path,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    session_service.touch_session(db, session)
    db.commit()
    db.refresh(document)
    logger.info(
        "document uploaded document_id=%s session_id=%s filename=%s size=%s",
        document.id, session_id, document.filename, document.file_size,
    )
    return document


def delete_document(
    db: DbSession, document_id, upload_dir: Path | None = None
) -> None:
    """Delete a document's record (cascades chunks) and its stored file."""
    document: Document | None = db.get(Document, document_id)
    if document is None:
        raise DocumentNotFoundError(f"Document {document_id} not found")
    storage.delete_uploaded_file(document.storage_path, upload_dir or settings.upload_dir)
    db.delete(document)
    db.commit()
    logger.info(
        "document deleted document_id=%s session_id=%s",
        document.id, document.session_id,
    )