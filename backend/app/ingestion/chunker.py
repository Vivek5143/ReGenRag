"""Document chunking.

Future responsibilities:
- splitting extracted text into chunks
- chunk size / overlap policy
- metadata creation per chunk (page, source, index)
"""


def chunk_text(*args, **kwargs):
    """Split raw document text into chunks.

    TODO (later phase): deterministic chunking (e.g. recursive character or
    semantic splitting) with per-chunk metadata. Not implemented in Phase 0.
    """
    raise NotImplementedError
