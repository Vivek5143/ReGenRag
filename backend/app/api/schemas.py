"""Pydantic request/response models for the API.

Response models are built explicitly from ORM objects so internal fields
(e.g. physical storage paths) are never leaked to clients.
"""

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from app.db.enums import DocumentStatus, SessionStatus
from app.retrieval.failure_classifier import RetrievalFailureCategory
from app.services.rag_service import HealingAction


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
    """Document metadata (never includes the internal storage path).

    ``chunk_count`` reports how many chunks were stored by ingestion
    (0 for an unprocessed or failed document).
    """

    document_id: uuid.UUID
    session_id: uuid.UUID
    filename: str
    status: DocumentStatus
    file_size: int
    chunk_count: int
    created_at: datetime
    processed_at: datetime | None


class QueryRequest(BaseModel):
    """A user question asked against a session's processed documents."""

    question: str

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        """Reject empty/whitespace-only questions and normalize the input."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class QuerySource(BaseModel):
    """A single retrieved chunk referenced as a source for an answer.

    ``score`` is the cosine similarity of the chunk to the query embedding
    (higher is more relevant).
    """

    document_id: uuid.UUID
    chunk_id: uuid.UUID
    chunk_index: int
    page_number: int | None
    score: float | None


class QueryRetrievalGrading(BaseModel):
    """Summary of how the retrieved evidence was graded.

    ``sufficient`` is True when at least one retrieved chunk met the similarity
    threshold; ``relevant_count``/``low_relevance_count`` break down the
    retrieved chunks. ``best_score`` and ``average_score`` are cosine
    similarities over all retrieved chunks (None when nothing was retrieved).
    """

    sufficient: bool
    threshold: float
    relevant_count: int
    low_relevance_count: int
    best_score: float | None
    average_score: float | None
    reason: str


class ChunkRelevanceResponse(BaseModel):
    """LLM relevance judgement for one retrieved chunk.

    ``relevant`` is the LLM's verdict on whether the chunk can help answer the
    question; ``reason`` is its one-sentence explanation. ``parse_failed``
    marks a verdict the grader could not parse (treated as not relevant).
    """

    chunk_id: uuid.UUID
    relevant: bool
    reason: str | None = None
    parse_failed: bool = False


class LlmRetrievalGradingResponse(BaseModel):
    """Aggregate LLM-based relevance grading for a query response.

    ``has_relevant_evidence`` is True when at least one chunk was judged
    semantically relevant by the LLM. This is informational; the similarity
    grader's sufficiency decision is unchanged.
    """

    relevant_count: int
    failed_count: int
    has_relevant_evidence: bool
    reason: str
    judgements: list[ChunkRelevanceResponse]


class HealingStepResponse(BaseModel):
    """Record of one healing action taken during the self-healing loop."""

    attempt: int
    action: HealingAction
    failure_category: RetrievalFailureCategory | None = None
    details: str = ""


class ErrorDetail(BaseModel):
    """Individual error detail for standardized error responses."""
    type: str
    message: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: float


class APIErrorResponse(BaseModel):
    """Standardized error response envelope for all API errors."""
    error: ErrorDetail


class APIMetadataResponse(BaseModel):
    """API metadata/versioning information."""
    name: str
    version: str
    description: Optional[str] = None
    environment: str


class QueryResponse(BaseModel):
    """Result of a RAG query: the generated answer plus its sources.

    ``retrieval_grading`` and ``llm_relevance`` are additive and informational;
    they report on retrieval quality and do not change the answer behavior.

    Phase 6 adds self-healing metadata:
    - ``healed``: True when recovery succeeded after an initial failure.
    - ``attempts``: Total attempts made (initial + retries).
    - ``healing_steps``: List of healing actions taken.
    - ``grounding_score``: Phase 5 grounding evaluation score [0, 1].
    - ``retry_exhausted``: True when all retries were used without success.
    """

    answer: str
    sources: list[QuerySource]
    retrieval_grading: QueryRetrievalGrading | None = None
    llm_relevance: LlmRetrievalGradingResponse | None = None
    failure_category: RetrievalFailureCategory | None = None
    rewritten_query: str | None = None
    healed: bool = False
    attempts: int = 1
    healing_steps: list[HealingStepResponse] = []
    grounding_score: float | None = None
    retry_exhausted: bool = False