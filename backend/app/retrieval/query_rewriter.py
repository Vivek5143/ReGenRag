"""Query rewriting for Phase 4.

When similarity grading deems retrieved evidence insufficient, this module can
ask the LLM to reformulate the user's query into a more retrieval-friendly
version. This step does NOT re-retrieve yet; it only produces a rewritten
query string for a later step.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.generation import llm
from app.retrieval.relevance_prompt import build_relevance_prompt
from app.retrieval.vector_store import RetrievedChunk
from app.retrieval.query_rewriter_prompt import build_query_rewrite_prompt

logger = get_logger(__name__)


def rewrite_query(
    original_query: str,
    *,
    chunks: list[RetrievedChunk],
    retrieval_grading,
    llm_relevance,
    client: llm.LLMClient | None = None,
) -> str | None:
    """Rewrite ``original_query`` to improve retrieval.

    Returns:
      - rewritten query string (non-empty) when the LLM produces usable text
      - None when the LLM output is empty/invalid or when the LLM call fails

    Notes:
      - This function never re-retrieves.
      - It uses deterministic, retrieval-oriented instructions in the prompt.
    """

    if not original_query or not original_query.strip():
        return None

    llm_client = client if client is not None else llm.get_llm_client()

    # Provide the model with a short summary of what was retrieved.
    context_snippet = "\n\n".join(
        f"[Source Page {c.page_number if c.page_number is not None else 'unknown'}]\n{c.content}"
        for c in chunks[: settings.rag_top_k]
    )

    prompt_text = build_query_rewrite_prompt(
        original_query=original_query,
        context_snippet=context_snippet,
    )

    try:
        raw = llm_client.complete(prompt_text, max_tokens=settings.llm_max_tokens)
    except LLMError:
        # Query rewriting is part of the self-healing pipeline.
        # Treat provider failures as insufficient evidence rather than
        # crashing the request.
        return None
    except Exception as exc:
        logger.error("query rewrite failed error=%s", exc, exc_info=True)
        return None

    rewritten = (raw or "").strip()
    if not rewritten:
        return None

    # Optional: if the model returned JSON, attempt to extract the string.
    # We intentionally keep this simple and robust.
    if rewritten.startswith("{") and "rewritten_query" in rewritten:
        # Avoid importing json + parsing in this minimal step unless needed.
        # If parsing fails, just return None.
        try:
            import json

            parsed = json.loads(rewritten)
            candidate = parsed.get("rewritten_query")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        except Exception:
            return None

    return rewritten
