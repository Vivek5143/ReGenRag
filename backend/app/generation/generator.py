"""Answer generation from a question and retrieved evidence.

Builds the baseline RAG prompt and sends it to the configured LLM client. This
is a plain single-shot generation — no grading, criticism, or retries (those
belong to later self-healing phases).
"""

from __future__ import annotations

from app.core.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.generation import llm, prompt

logger = get_logger(__name__)


def generate_answer(
    question: str,
    context: str,
    *,
    client: llm.LLMClient | None = None,
) -> str:
    """Generate a grounded answer for ``question`` given ``context``.

    Returns the raw answer text. Any provider failure surfaces as ``LLMError``;
    an explicitly supplied ``client`` (used by tests) bypasses provider config.
    """
    prompt_text = prompt.build_baseline_prompt(question, context)
    llm_client = client if client is not None else llm.get_llm_client()

    try:
        answer = llm_client.complete(prompt_text, max_tokens=settings.llm_max_tokens)
    except LLMError:
        raise
    except Exception as exc:
        logger.error("llm call failed error=%s", exc, exc_info=True)
        raise LLMError(f"LLM call failed: {exc}") from exc

    if not answer:
        raise LLMError("LLM returned an empty answer")

    logger.info("answer generated chars=%s", len(answer))
    return answer
