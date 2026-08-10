"""Document chunking.

Splits cleaned page text into deterministic, character-based chunks while
preserving page metadata. Chunks never span pages, so every chunk carries a
single, exact ``page_number`` for later citation. ``chunk_index`` is assigned
globally across the document (sequential, deterministic). Boundaries are chosen
at paragraph / sentence breaks when a good one exists near the window size,
falling back to a hard character split otherwise.
"""

from dataclasses import dataclass

from app.ingestion.loader import LoadedPage


@dataclass
class Chunk:
    """A single chunk: sequential index, content, page, and source metadata."""

    chunk_index: int
    content: str
    page_number: int
    metadata: dict


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split one block of text into overlapping pieces of at most ``chunk_size``.

    Prefers splitting at paragraph (``\\n\\n``) then line (``\\n``) then sentence
    boundaries near the window end; otherwise splits at ``chunk_size``. Returns
    cleaned (stripped) pieces. Empty input yields an empty list.
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Search for a natural split inside the back half of the window so
            # pieces stay reasonably full while avoiding tiny stragglers.
            window_start = end - chunk_size // 2
            boundary = _find_boundary(text, window_start, end)
            if boundary is not None:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        # Advance by the stride, keeping the tail of this piece as overlap.
        start = end - chunk_overlap if chunk_overlap > 0 else end
        if start <= 0:  # guard against a pathological overlap never progressing
            break
    return pieces


def _find_boundary(text: str, low: int, high: int) -> int | None:
    """Return the best boundary index within ``[low, high)``, or None."""
    for marker in ("\n\n", "\n", ". ", "! ", "? "):
        idx = text.rfind(marker, low, high)
        if idx != -1:
            return idx + len(marker)
    return None


def chunk_pages(
    pages: list[LoadedPage],
    *,
    document_id,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk a list of loaded pages into document-scoped ``Chunk`` objects.

    Each page is chunked independently (chunks never cross pages). Metadata on
    every chunk records ``document_id``, ``filename``, ``page_number``,
    ``chunk_index``, and the page's ``source``.
    """
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for piece in chunk_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_index=index,
                    content=piece,
                    page_number=page.page_number,
                    metadata={
                        **page.metadata,
                        "document_id": str(document_id),
                        "filename": filename,
                        "page_number": page.page_number,
                        "chunk_index": index,
                    },
                )
            )
            index += 1
    return chunks