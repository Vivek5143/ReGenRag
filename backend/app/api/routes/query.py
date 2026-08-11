"""RAG query endpoint: ask a question against a session's documents."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DbSession

from app.api.schemas import QueryRequest, QueryResponse, QuerySource
from app.core.exceptions import (
    LLMConfigError,
    LLMError,
    NoProcessedDocumentError,
    ReGenRAGError,
    RetrievalError,
    SessionNotFoundError,
)
from app.db.database import get_db
from app.services import rag_service

router = APIRouter(prefix="/sessions/{session_id}", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the session's processed documents",
)
def query_session(
    session_id: uuid.UUID,
    payload: QueryRequest,
    db: DbSession = Depends(get_db),
):
    try:
        result = rag_service.answer_question(db, session_id, payload.question)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except NoProcessedDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except LLMConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    except ReGenRAGError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    return QueryResponse(
        answer=result.answer,
        sources=[
            QuerySource(
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                chunk_index=source.chunk_index,
                page_number=source.page_number,
                score=source.score,
            )
            for source in result.sources
        ],
    )
