"""Baseline RAG evaluation runner.

Executes the original Phase‑3 RAG pipeline without any grading, query
rewriting, or self‑healing. It simply:

1. Retrieves the top‑K chunks for the case's query.
2. Generates an answer from those chunks.
3. Computes deterministic retrieval metrics (HR, MRR, nDCG).

The result is a :class:`~app.evaluation.result.EvalResult` that contains
retrieval metrics and a placeholder ``grounding_score`` / ``answer_quality``
of ``None`` (those need an LLM evaluator, which this runner deliberately
avoids).

Usage
-----
The runner is designed to be called from the ``comparator`` or the CLI entry
point. It requires a SQLAlchemy ``DbSession`` connected to a session that has
processed documents (chunks) so the retriever can find evidence.

Example
-------
::

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as DbSession
    from app.evaluation.baseline_runner import run_baseline
    from app.evaluation.model import EvalCase

    engine = create_engine("postgresql+psycopg2://.../regenrag")
    with DbSession(engine) as db:
        case = EvalCase(id="case-1", query="What is the capital of France?")
        result = run_baseline(db, case)
        # result.hit_rate, result.mrr, result.ndcg are now populated
"""
from __future__ import annotations

import uuid
from sqlalchemy.orm import Session as DbSession

from ..core.config import settings
from .result import EvalResult
from .model import EvalCase
from ..generation.generator import generate_answer
from ..retrieval import retriever
from ..retrieval.vector_store import RetrievedChunk
from .metrics import compute_retrieval_metrics


def run_baseline(db: DbSession, case: EvalCase) -> EvalResult:
    """Run the Phase‑3 baseline RAG pipeline on *case*.

    The pipeline steps are:

    1. **Retrieve** – call ``retriever.retrieve`` with the configured ``top_k``.
    2. **Generate** – call ``generator.generate_answer`` with the retrieved
       chunks' text joined as context.
    3. **Metric extraction** – compute HR@K, MRR, nDCG using the already‑relevant
       chunk IDs from ``case.relevant_chunk_ids`` (oracle signal).

    The retrieval result is an empty list when the session has no searchable
    chunks; in that case all metrics are ``0.0`` and the answer is a
    deterministic "no-chunks" placeholder.

    Returns
    -------
    EvalResult
        A dataclass populated with retrieval metrics and generation metadata.
    """
    # ── 1. Retrieve chunks ────────────────────────────────────────────
    top_k = settings.rag_top_k
    chunks: list[RetrievedChunk] = retriever.retrieve(
        db,
        session_id=uuid.UUID(case.id),
        question=case.query,
        top_k=top_k,
    )

    # ── 2. Generate answer ────────────────────────────────────────────
    if chunks:
        context = "\n\n".join(chunk.content for chunk in chunks)
        answer = generate_answer(case.query, context)
    else:
        # Deterministic placeholder when no evidence is found.
        answer = "I couldn't find enough relevant information in the provided documents to answer this question reliably."

    # ── 3. Compute retrieval metrics ──────────────────────────────────
    relevant_ids: set[uuid.UUID] = set()
    if case.relevant_chunk_ids is not None:
        relevant_ids = {
            uuid.UUID(c) if isinstance(c, str) else c
            for c in case.relevant_chunk_ids
        }

    metrics = compute_retrieval_metrics(chunks, relevant_ids, k=top_k)

    return EvalResult(
        case_id=case.id,
        query=case.query,
        hit_rate=metrics["hit_rate"],
        mrr=metrics["mrr"],
        ndcg=metrics["ndcg"],
        grounding_score=None,  # Phase 5 grounding not part of baseline
        answer_quality=None,   # LLM-based correctness not part of baseline
        failure_category=None,
        rewritten_query=None,
        expected_answer=case.expected_answer,
    )