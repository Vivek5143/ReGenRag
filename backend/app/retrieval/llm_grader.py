"""LLM-based retrieval relevance grading (Phase 4, step 2).

For each retrieved chunk, a configurable LLM judges whether the chunk actually
contains information relevant to answering the user's question — semantic
relevance, distinct from the similarity score the retriever computed. Each
verdict is a machine-readable JSON object parsed into a ``ChunkRelevance``;
results are aggregated into ``LlmRetrievalGrading``.

No sufficiency decision is made here: the similarity-based rule from step 1 is
unchanged. This step only adds semantic relevance information. Query
rewriting, re-retrieval, and retries are later steps.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.generation import llm
from app.retrieval.relevance_prompt import build_relevance_prompt
from app.retrieval.vector_store import RetrievedChunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChunkRelevance:
    """Semantic relevance judgement for a single retrieved chunk.

    ``relevant`` is the LLM's verdict. ``reason`` is the LLM's one-sentence
    explanation (None when it could not be parsed). ``parse_failed`` is True
    when the LLM output was not a valid JSON verdict, in which case ``relevant``
    is conservatively False.
    """

    chunk_id: uuid.UUID
    relevant: bool
    reason: str | None = None
    parse_failed: bool = False


@dataclass(frozen=True)
class LlmRetrievalGrading:
    """Aggregate LLM-based relevance grading over the retrieved chunks.

    ``judgements`` holds one entry per retrieved chunk, in retrieval order.
    ``has_relevant_evidence`` is True when at least one chunk was judged
    relevant. This is informational only; it does not replace the similarity
    grader's sufficiency rule.
    """

    judgements: list[ChunkRelevance] = field(default_factory=list)
    relevant_count: int = 0
    failed_count: int = 0
    has_relevant_evidence: bool = False
    reason: str = ""


def _extract_json_object(text: str) -> dict | None:
    """Return the balanced JSON object in ``text``, or None.

    Accepts responses wrapped in prose or markdown code fences by slicing from
    the first ``{`` to the last ``}``.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_bool(value: object) -> bool | None:
    """Coerce a parsed JSON value to a strict bool, or None when ambiguous."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "1"):
            return True
        if normalized in ("false", "no", "0"):
            return False
    return None


def _parse_judgement(chunk_id: uuid.UUID, raw: str) -> ChunkRelevance:
    """Parse one LLM response into a ``ChunkRelevance``, never raising."""
    parsed = _extract_json_object(raw)
    if parsed is None:
        logger.warning(
            "llm relevance output unparseable chunk_id=%s output=%r",
            chunk_id, raw[:200],
        )
        return ChunkRelevance(chunk_id=chunk_id, relevant=False, parse_failed=True)

    relevant = _coerce_bool(parsed.get("relevant"))
    if relevant is None:
        logger.warning(
            "llm relevance output missing valid 'relevant' chunk_id=%s output=%r",
            chunk_id, raw[:200],
        )
        return ChunkRelevance(chunk_id=chunk_id, relevant=False, parse_failed=True)

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = None
    return ChunkRelevance(chunk_id=chunk_id, relevant=relevant, reason=reason)


def grade_relevance(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    client: llm.LLMClient | None = None,
) -> LlmRetrievalGrading:
    """Judge, per chunk, whether it is semantically relevant to ``question``.

    One LLM call is made per chunk, in retrieval order, requesting a
    machine-readable JSON verdict. Empty ``chunks`` makes no LLM calls. An
    explicitly supplied ``client`` (used by tests) bypasses provider config;
    otherwise the shared cached client is used. Provider failures surface as
    ``LLMError`` (unexpected exceptions are wrapped), matching the generator.
    """
    llm_client = client if client is not None else llm.get_llm_client()

    judgements: list[ChunkRelevance] = []
    for chunk in chunks:
        prompt_text = build_relevance_prompt(question, chunk.content)
        try:
            raw = llm_client.complete(
                prompt_text, max_tokens=settings.llm_max_tokens
            )
        except LLMError:
            raise
        except Exception as exc:
            logger.error(
                "llm relevance grading call failed chunk_id=%s error=%s",
                chunk.chunk_id, exc, exc_info=True,
            )
            raise LLMError(f"LLM relevance grading failed: {exc}") from exc
        judgements.append(_parse_judgement(chunk.chunk_id, raw))

    relevant_count = sum(1 for j in judgements if j.relevant)
    failed_count = sum(1 for j in judgements if j.parse_failed)

    if not judgements:
        reason = "no chunks to grade"
    else:
        reason = (
            f"{relevant_count} of {len(judgements)} chunks judged relevant"
            + (f", {failed_count} could not be parsed" if failed_count else "")
        )

    logger.info(
        "llm relevance graded chunks=%s relevant=%s failed=%s",
        len(judgements), relevant_count, failed_count,
    )

    return LlmRetrievalGrading(
        judgements=judgements,
        relevant_count=relevant_count,
        failed_count=failed_count,
        has_relevant_evidence=relevant_count > 0,
        reason=reason,
    )
