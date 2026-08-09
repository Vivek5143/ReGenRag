"""Filesystem helpers for session-scoped uploads.

Uploads live under ``<upload_dir>/<session_id>/<document_id>.pdf``. All access
resolves stored relative paths against a configured root and rejects path
traversal. The internal storage layout is never exposed to clients.
"""

from pathlib import Path

from app.core.exceptions import DocumentValidationError


def resolve_upload_path(relative_path: str, upload_dir: Path | str) -> Path:
    """Resolve a stored relative path inside the upload root; reject traversal."""
    root = Path(upload_dir).resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise DocumentValidationError(f"Invalid storage path: {relative_path}")
    return candidate


def delete_uploaded_file(relative_path: str, upload_dir: Path | str) -> bool:
    """Delete an uploaded file if it exists. Idempotent: returns deleted or not."""
    path = resolve_upload_path(relative_path, upload_dir)
    if path.exists():
        path.unlink()
        return True
    return False