"""Retrieval: embed the query and fetch top-k relevant chunks per session.

Reuses the same local embedding model used for ingestion (all-MiniLM-L6-v2) so
queries live in the same 384-dim space as stored document chunks. Ranked search
itself is delegated to ``vector_store.similarity_search`` (pgvector).
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session as DbSession

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion import embedder
from app.retrieval import vector_store

logger = get_logger(__name__)


def embed_query(question: str) -> list[float]:
    """Embed a single query into the same normalized space as document chunks."""
    vectors = embedder.embed_texts([question])
    return vectors[0]


def retrieve(
    db: DbSession,
    session_id: uuid.UUID,
    question: str,
    top_k: int | None = None,
) -> list[vector_store.RetrievedChunk]:
    """Return the ``top_k`` most relevant chunks for ``question`` in a session.

    ``top_k`` defaults to the configured ``RAG_TOP_K``. The query is embedded
    with the shared local model, then ranked against the session's chunks via
    pgvector cosine similarity. Returns an empty list when the session has no
    searchable chunks.
    """
    k = top_k if top_k is not None else settings.rag_top_k
    query_embedding = embed_query(question)
    chunks = vector_store.similarity_search(
        db, query_embedding, session_id=session_id, top_k=k
    )
    logger.info(
        "retrieval completed session_id=%s k=%s hits=%s",
        session_id, k, len(chunks),
    )
    return chunks
