"""Retrieval.

Future responsibilities:
- embedding the user's query
- top-k similarity search against the session's chunks (RAG_TOP_K)
- returning ranked evidence documents with scores
- exposing signal (e.g. score) that later phases use to grade retrieval quality
"""


def retrieve(*args, **kwargs):
    """Retrieve the most relevant chunks for a query within a session.

    TODO (later phase): embed query -> vector search -> ranked evidence.
    Not implemented in Phase 0.
    """
    raise NotImplementedError
