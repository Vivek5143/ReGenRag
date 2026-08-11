"""Vector store: pgvector similarity search over session-scoped chunks.

This module is the only place that talks to pgvector; it owns the SQLAlchemy
query that ranks ``document_chunks`` by cosine distance against a query
embedding. Embedding the query and interpreting the results lives in
``retriever.py``.

Cosine distance (``<=>``) is used because Phase 2 stores L2-normalized
embeddings, so distance is monotonically equivalent to ``1 - cosine``
similarity. Higher similarity means a more relevant chunk.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk


class RetrievedChunk:
    """One chunk returned by similarity search with its ranking metadata.

    ``score`` is cosine similarity to the query (1 - cosine distance), so a
    higher score means the chunk is more relevant.
    """

    __slots__ = ("chunk_id", "document_id", "chunk_index", "page_number", "content", "score")

    def __init__(
        self,
        *,
        chunk_id: uuid.UUID,
        document_id: uuid.UUID,
        chunk_index: int,
        page_number: int | None,
        content: str,
        score: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.chunk_index = chunk_index
        self.page_number = page_number
        self.content = content
        self.score = score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RetrievedChunk):
            return NotImplemented
        return (
            self.chunk_id == other.chunk_id
            and self.document_id == other.document_id
            and self.chunk_index == other.chunk_index
            and self.page_number == other.page_number
            and self.content == other.content
            and self.score == other.score
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RetrievedChunk(chunk_id={self.chunk_id}, document_id={self.document_id}, "
            f"chunk_index={self.chunk_index}, page_number={self.page_number}, "
            f"score={self.score:.4f})"
        )


def similarity_search(
    db: DbSession,
    query_embedding: list[float],
    *,
    session_id: uuid.UUID,
    top_k: int,
) -> list[RetrievedChunk]:
    """Return the ``top_k`` chunks most similar to ``query_embedding``.

    Results are scoped to chunks belonging to documents in ``session_id`` and
    ordered by ascending cosine distance (most similar first). Only chunks with
    a stored embedding are considered. Returns an empty list when the session
    has no searchable chunks.
    """
    # Pass the raw query vector (a list) so the `<=>` operator binds it with the
    # column's Vector type and returns a real cosine distance (float). Wrapping
    # it in `Vector(...)` passes a Vector type instance as the operand instead,
    # which breaks the returned value (float() fails on it).
    distance_col = DocumentChunk.embedding.cosine_distance(query_embedding)

    rows = db.execute(
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.page_number,
            DocumentChunk.content,
            distance_col.label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.session_id == session_id,
            DocumentChunk.embedding.is_not(None),
        )
        .order_by(distance_col)
        .limit(top_k)
    ).all()

    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_index=row.chunk_index,
            page_number=row.page_number,
            content=row.content,
            score=1.0 - float(row.distance),
        )
        for row in rows
    ]
