"""Document loading.

Future responsibilities:
- PDF loading
- text extraction
- page preservation
- validation against MAX_FILE_SIZE_MB
- attachment of source metadata (filename, page numbers)
"""


def load_document(*args, **kwargs):
    """Load a document for a session.

    TODO (later phase): parse PDF(s), extract text while preserving page
    boundaries, and return raw text with metadata. Not implemented in Phase 0.
    """
    raise NotImplementedError
