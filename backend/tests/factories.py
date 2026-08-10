"""Test fixtures for generating real PDFs.

reportlab is a test-only dependency used to build valid, text-based PDFs so the
loader and ingestion tests exercise real parsing rather than stubbed bytes.

Public helpers:
    - ``pdf_bytes(*pages, title=...)`` -> bytes of a PDF, one page per ``pages``
      string. An empty string yields one blank page.
    - ``pdf_bytes()`` with no pages -> a zero-page PDF.
"""


def pdf_bytes(*pages: str, title: str | None = None) -> bytes:
    """Build a text-based PDF in memory. Each ``pages`` entry becomes a page."""
    from io import BytesIO

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    if title:
        c.drawString(72, 770, title[:80])
    for text in pages:
        line_y = 750
        for line in text.split("\n"):
            # Truncate long lines so they stay on the (letter) page; pypdf
            # extracts the text regardless, but keeping it tidy avoids overlap.
            c.drawString(72, line_y, line[:95])
            line_y -= 14
        c.showPage()
    c.save()
    return buffer.getvalue()