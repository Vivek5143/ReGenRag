"""Document loading: extract page-aware text from an uploaded PDF.

Uses pypdf, a lightweight pure-Python parser, to open a PDF and pull text out
page by page. Page boundaries are preserved so downstream phases can cite
specific pages. Empty pages are retained (page numbers stay accurate); a file
that cannot be parsed at all is rejected with a ``DocumentProcessingError``.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import DocumentProcessingError


@dataclass
class LoadedPage:
    """One extracted page: 1-based number, raw text, and source metadata."""

    page_number: int
    text: str
    metadata: dict


def load_document(path: Path, source: str) -> list[LoadedPage]:
    """Extract page-aware text from the PDF at ``path``.

    ``source`` is the human-facing filename recorded in each page's metadata.
    Returns one ``LoadedPage`` per physical page, in order. Raises
    ``DocumentProcessingError`` for files that cannot be opened or parsed.
    """
    pages: list[LoadedPage] = []
    # pypdf reads pages lazily, so the file handle must stay open while we
    # iterate and extract text from every page.
    try:
        with Path(path).open("rb") as file:
            reader = PdfReader(file)
            for index, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    raise DocumentProcessingError(
                        f"Failed to extract text from page {index} of "
                        f"'{source}': {exc}"
                    ) from exc
                pages.append(
                    LoadedPage(
                        page_number=index,
                        text=text,
                        metadata={"source": source},
                    )
                )
    except DocumentProcessingError:
        raise
    except Exception as exc:  # pypdf raises on malformed/empty/corrupt files
        raise DocumentProcessingError(
            f"Unable to open or parse PDF '{source}': {exc}"
        ) from exc

    if not pages:
        raise DocumentProcessingError(f"PDF '{source}' contains no pages")
    return pages
