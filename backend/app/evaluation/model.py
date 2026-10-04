"""Evaluation data model.

An evaluation case represents a single offline question we want to measure the
RAG system on. All fields are validated with Pydantic; required fields are
checked eagerly so a malformed case surface as a ``ValueError`` before any
metric computation runs.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class EvalCase(BaseModel):
    """One item in an offline evaluation set.

    ``relevant_chunk_ids`` is the authoritative oracle signal for retrieval
    metrics (hit rate, MRR, nDCG). If it is present, chunk IDs must be unique
    and valid UUIDs. ``relevant_document_ids`` is an optional document-level
    supplement that is ignored by the current deterministic metrics but may
    drive future document-level grading.

    ``expected_answer`` is a human-authored reference text used by answer
    metrics (grounding / faithfulness) once those are implemented. For pure
    retrieval metrics the field is unused but present to make the case schema
    complete.
    """

    id: str
    query: str
    expected_answer: Optional[str] = None
    relevant_document_ids: list[uuid.UUID] | None = None
    relevant_chunk_ids: list[uuid.UUID] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("id must not be blank")
        return stripped

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped

    @field_validator("relevant_chunk_ids", mode="before")
    @classmethod
    def _normalize_chunk_ids(
        cls, value: object
    ) -> list[uuid.UUID] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [cls._coerce_uuid(x) for x in value]
        raise ValueError("relevant_chunk_ids must be a list of UUIDs")

    @field_validator("relevant_document_ids", mode="before")
    @classmethod
    def _normalize_doc_ids(
        cls, value: object
    ) -> list[uuid.UUID] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [cls._coerce_uuid(x) for x in value]
        raise ValueError("relevant_document_ids must be a list of UUIDs")

    @staticmethod
    def _coerce_uuid(value: object) -> uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value
        if isinstance(value, str):
            return uuid.UUID(value)
        raise ValueError(f"expected a UUID, got {type(value).__name__}")
