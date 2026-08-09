"""Session endpoints: create, inspect, close."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DbSession

from app.api.schemas import SessionCreateResponse, SessionResponse
from app.core.exceptions import SessionNotFoundError
from app.db.database import get_db
from app.services import cleanup_service, session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post(
    "",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new temporary document session",
)
def create_session(db: DbSession = Depends(get_db)):
    session = session_service.create_session(db)
    return SessionCreateResponse(
        session_id=session.id,
        status=session.status,
        expires_at=session.expires_at,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get session metadata",
)
def get_session(session_id: uuid.UUID, db: DbSession = Depends(get_db)):
    try:
        session = session_service.get_session(db, session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return SessionResponse(
        session_id=session.id,
        status=session.status,
        created_at=session.created_at,
        last_activity=session.last_activity,
        expires_at=session.expires_at,
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Close a session and remove its documents",
)
def close_session(session_id: uuid.UUID, db: DbSession = Depends(get_db)):
    try:
        session_service.close_session(db, session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    # Close marks the session CLOSED; cleanup removes its contents and files.
    cleanup_service.cleanup_session(db, session_id)