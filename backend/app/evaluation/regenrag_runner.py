"""ReGenRAG (Phase 6) evaluation runner.

This runner executes the current Phase‑6 ReGenRAG pipeline against an evaluation
case and returns the same schema as the baseline runner so that the ``comparator``
can produce a side‑by‑side metric table.

What it does
------------
1. Calls the public ``rag_service.answer_question`` entry point, which internally
   runs the self‑healing loop (query‑rewrite → re‑retrieve → re‑grade → grounding
   → retry).
2. Captures every artefact the service produces:
   - initial retrieval (chunks + scores)
   - retrieval‑grading result (sufficient / threshold)
   - failure category that triggered the healing path
   - query rewrite text (if any)
   - re‑retrieval chunks (after rewrite)
   - grounding score from Phase 5 critic
   - total number of attempts / healing steps
   - whether the final answer was recovered
3. Returns an ``EvalResult`` populated with all those fields, mirroring the
   ``RagAnswer`` model but in the Phase‑7 evaluation shape.

The runner deliberately uses the **same** service that the production API uses,
so no duplication of retrieval/generation logic is needed. It only records the
outcome for later comparison with the baseline.

Usage
-----
The runner is called from the ``comparator`` or the CLI entry point. It
requires a SQLAlchemy ``DbSession`` connected to a session that has processed
documents.

Example
-------
::

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as DbSession
    from app.evaluation.regenrag_runner import run_regenrag
    from app.evaluation.model import EvalCase

    engine = create_engine("postgresql+psycopg2://.../regenrag")
    with DbSession(engine) as db:
        case = EvalCase(id="case-1", query="What is the capital of France?")
        result = run_regenrag(db, case)
        # result.grounding_score, result.healed, result.attempts, etc. are populated
"""
from __future__ import annotations

import uuid
from sqlalchemy.orm import Session as DbSession

from ..core.config import settings
from .result import EvalResult
from .model import EvalCase
from ..services.rag_service import answer_question
from ..retrieval.vector_store import RetrievedChunk
from .metrics import compute_retrieval_metrics


def run_regenrag(db: DbSession, case: EvalCase) -> EvalResult:
    """Run the Phase‑6 ReGenRAG pipeline on *case*.

    This invokes the production ``rag_service.answer_question`` endpoint, which
    implements the full self‑healing loop (retrieval grading, query rewrite,
    re‑retrieval, grounding critic, bounded retries).  All metadata that the
    service produces is captured and translated into an ``EvalResult`` so the
    comparator can compare it directly with the baseline result for the same
    case.

    Steps performed by the underlying service (Phase 6):
    1. Retrieve evidence for the current query.
    2. Grade retrieval (similarity + LLM relevance).
    3. If insufficient: rewrite the query and re‑retrieve (bounded by
       ``settings.max_rag_retries``).
    4. If sufficient: build context, generate answer.
    5. Grade grounding (Phase 5).
    6. If grounding fails: improve query/context, re‑retrieve, regenerate
       (bounded).
    7. Return the final answer with healing metadata.

    The runner simply delegates to the service and maps the returned
    ``RagAnswer`` fields onto the ``EvalResult`` schema.

    Returns
    -------
    EvalResult
        A dataclass populated with retrieval metrics, grounding score, healing
        metadata, and the final answer.
    """
    rag_answer = answer_question(
        db,
        session_id=uuid.UUID(case.id),
        question=case.query,
    )

    relevant_ids: set[uuid.UUID] = set()
    if case.relevant_chunk_ids is not None:
        relevant_ids = {
            uuid.UUID(c) if isinstance(c, str) else c
            for c in case.relevant_chunk_ids
        }

    retrieved: list[RetrievedChunk] = [
        RetrievedChunk(
            chunk_id=s.chunk_id,
            document_id=s.document_id,
            chunk_index=s.chunk_index,
            page_number=s.page_number,
            content="",
            score=s.score,
        )
        for s in rag_answer.sources
    ]

    metrics = compute_retrieval_metrics(retrieved, relevant_ids, k=settings.rag_top_k)

    # Create EvalResult with all fields, plus healing metadata
    result = EvalResult(
        case_id=case.id,
        query=case.query,
        hit_rate=metrics["hit_rate"],
        mrr=metrics["mrr"],
        ndcg=metrics["ndcg"],
        grounding_score=rag_answer.grounding_score,
        answer_quality=None,
        failure_category=rag_answer.failure_category,
        rewritten_query=rag_answer.rewritten_query,
        expected_answer=case.expected_answer,
        attempts=rag_answer.attempts,
        healing_steps=rag_answer.healing_steps,
        retry_exhausted=rag_answer.retry_exhausted,
    )

    return result