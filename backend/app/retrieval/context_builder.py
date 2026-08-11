"""Context building: turn retrieved chunks into LLM-readable evidence.

Chunks keep their page number (and retrieval order), so the LLM can see exactly
which page each piece of evidence came from and cite it later. The output is
deterministic: same chunks in, same context out.
"""

from __future__ import annotations

from app.retrieval.vector_store import RetrievedChunk

_UNKNOWN_PAGE = "unknown"


def _source_line(page_number: int | None) -> str:
    """Render a ``[Source: Page N]`` marker, handling a missing page number."""
    page = page_number if page_number is not None else _UNKNOWN_PAGE
    return f"[Source: Page {page}]"


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format ``chunks`` into a single evidence context block.

    Each chunk becomes ``[Source: Page N]\\n<chunk text>`` and blocks are
    separated by a blank line. Empty input yields an empty string, which callers
    treat as "no evidence retrieved".
    """
    if not chunks:
        return ""
    blocks = [f"{_source_line(chunk.page_number)}\n{chunk.content}" for chunk in chunks]
    return "\n\n".join(blocks)
