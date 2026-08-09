"""Document endpoints: upload and inspect documents within a session."""

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import selectinload

from app.api.schemas import DocumentResponse
from app.core.exceptions import (
    DocumentValidationError,
    SessionNotActiveError,
    SessionNotFoundError,
)
from app.db import models
from app.db.database import get_db
from app.services import document_service

router = APIRouter(prefix="/sessions/{session_id}/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF to a session",
)
def upload_document(
    session_id: uuid.UUID,
    file: UploadFile = File(...),
    db: DbSession = Depends(get_db),
):
    try:
        document = document_service.upload_document(db, session_id, file)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except SessionNotActiveError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return _to_response(document)


@router.get(
    "",
    response_model=list[DocumentResponse],
    summary="List documents in a session",
)
def list_documents(session_id: uuid.UUID, db: DbSession = Depends(get_db)):
    # Ensure the session exists (404 otherwise).
    session = db.get(models.Session, session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    documents = db.scalars(
        select(models.Document)
        .where(models.Document.session_id == session_id)
        .options(selectinload(models.Document.chunks))
    ).all()
    return [_to_response(doc) for doc in documents]


def _to_response(doc) -> DocumentResponse:
    return DocumentResponse(
        document_id=doc.id,
        session_id=doc.session_id,
        filename=doc.filename,
        status=doc.status,
        file_size=doc.file_size,
        created_at=doc.created_at,
        processed_at=doc.processed_at,
    )