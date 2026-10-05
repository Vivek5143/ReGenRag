"""RAG query endpoint: ask a question against a session's documents."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DbSession

from app.api.schemas import (
    ChunkRelevanceResponse,
    HealingStepResponse,
    LlmRetrievalGradingResponse,
    QueryRequest,
    QueryResponse,
    QueryRetrievalGrading,
    QuerySource,
)
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

    grading = result.retrieval_grading
    relevance = result.llm_relevance
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
        retrieval_grading=(
            QueryRetrievalGrading(
                sufficient=grading.sufficient,
                threshold=grading.threshold,
                relevant_count=len(grading.relevant_chunks),
                low_relevance_count=len(grading.low_relevance_chunks),
                best_score=grading.best_score,
                average_score=grading.average_score,
                reason=grading.reason,
            )
            if grading is not None
            else None
        ),
        failure_category=result.failure_category,
        rewritten_query=result.rewritten_query,
        llm_relevance=(
            LlmRetrievalGradingResponse(
                relevant_count=relevance.relevant_count,
                failed_count=relevance.failed_count,
                has_relevant_evidence=relevance.has_relevant_evidence,
                reason=relevance.reason,
                judgements=[
                    ChunkRelevanceResponse(
                        chunk_id=judgement.chunk_id,
                        relevant=judgement.relevant,
                        reason=judgement.reason,
                        parse_failed=judgement.parse_failed,
                    )
                    for judgement in relevance.judgements
                ],
            )
            if relevance is not None
            else None
        ),
        healed=result.healed,
        attempts=result.attempts,
        healing_steps=[
            HealingStepResponse(
                attempt=step.attempt,
                action=step.action,
                failure_category=step.failure_category,
                details=step.details,
            )
            for step in result.healing_steps
        ],
        grounding_score=result.grounding_score,
        retry_exhausted=result.retry_exhausted,
    )
