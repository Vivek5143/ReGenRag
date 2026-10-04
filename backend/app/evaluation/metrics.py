"""Deterministic retrieval metrics.

Measures how well a ranked retrieval result matches an oracle list of relevant
chunk IDs. Pure functions — no LLM, no network, no database, no configuration.
All metrics are defined on ``RetrievedChunk`` objects (``chunk_id`` and ``score``
are the only fields used; ``content``/``page_number`` are ignored).

Metric definitions and assumptions
----------------------------------

**Hit Rate (HR@K)**
    Binary: ``1`` if at least one retrieved chunk is in the relevant set, else
    ``0``. The "K" is implicit — it equals the length of the retrieved list
    (or ``top_k`` when the caller restricts to a prefix). An empty retrieval
    returns ``0``.

**Mean Reciprocal Rank (MRR)**
    ``1 / r`` where ``r`` is the rank (1-based position) of the first
    retrieved chunk whose ``chunk_id`` is in the relevant set. If no chunk
    matches, returns ``0``. An empty retrieval returns ``0``.

**Normalized Discounted Cumulative Gain (nDCG@K)**
    Uses **binary gain**: each retrieved chunk scores ``1`` if its
    ``chunk_id`` is in the relevant set, else ``0``. DCG is computed over the
    first ``K`` results (or the full list if shorter than ``K``); the ideal
    DCG is computed with the same ``K`` cap against the total number of
    relevant chunks in the oracle.

    Assumptions:
    - Each chunk is either relevant or irrelevant (no graded relevance yet).
    - ``K`` defaults to the length of the retrieved list so the metric is
      well-defined even when the list is shorter than the oracle size.
    - When there are no relevant chunks in the oracle, nDCG is ``0`` (not
      undefined); this avoids division-by-zero and keeps the metric safe
      for queries where no ground truth is available.
    - An empty retrieval always returns ``0`` regardless of the oracle.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable

from app.retrieval.vector_store import RetrievedChunk


def hit_rate(
    retrieved: list[RetrievedChunk],
    relevant_chunk_ids: set[uuid.UUID],
) -> float:
    """Return ``1.0`` when any retrieved chunk is in ``relevant_chunk_ids``."""

    if not retrieved:
        return 0.0
    return 1.0 if any(c.chunk_id in relevant_chunk_ids for c in retrieved) else 0.0


def mean_reciprocal_rank(
    retrieved: list[RetrievedChunk],
    relevant_chunk_ids: set[uuid.UUID],
) -> float:
    """Return ``1/r`` for the first relevant chunk at rank ``r``, else ``0``."""

    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def _dcg(scores: list[float]) -> float:
    """Discounted cumulative gain over a list of relevance scores."""

    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def normalized_dcg(
    retrieved: list[RetrievedChunk],
    relevant_chunk_ids: set[uuid.UUID],
    *,
    k: int | None = None,
) -> float:
    """nDCG with binary relevance and an optional cap at position ``k``.

    When ``k`` is omitted, it defaults to ``len(retrieved)`` so the metric
    is always defined (never raises on empty inputs).
    """

    if not retrieved or not relevant_chunk_ids:
        return 0.0

    cap = k if k is not None else len(retrieved)
    cap = max(1, min(cap, len(retrieved)))

    gains = [1.0 if c.chunk_id in relevant_chunk_ids else 0.0 for c in retrieved[:cap]]
    dcg = _dcg(gains)

    # Ideal DCG: all relevant chunks packed at the top, capped at ``cap``.
    ideal_gains = [1.0] * min(len(relevant_chunk_ids), cap) + [0.0] * max(
        0, cap - len(relevant_chunk_ids)
    )
    idcg = _dcg(ideal_gains)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    retrieved: list[RetrievedChunk],
    relevant_chunk_ids: Iterable[str | uuid.UUID],
    *,
    k: int | None = None,
) -> dict[str, float]:
    """Compute hit rate, MRR, and nDCG for a single retrieval result.

    ``relevant_chunk_ids`` may contain strings or UUIDs; both are accepted and
    normalized internally. Returns a flat dict keyed by metric name:

        {"hit_rate": float, "mrr": float, "ndcg": float}
    """

    ids: set[uuid.UUID] = set()
    for ref in relevant_chunk_ids:
        if isinstance(ref, str):
            ids.add(uuid.UUID(ref))
        elif isinstance(ref, uuid.UUID):
            ids.add(ref)
        else:
            raise TypeError(
                f"relevant_chunk_ids must contain str or UUID, got {type(ref).__name__}"
            )

    return {
        "hit_rate": hit_rate(retrieved, ids),
        "mrr": mean_reciprocal_rank(retrieved, ids),
        "ndcg": normalized_dcg(retrieved, ids, k=k),
    }
