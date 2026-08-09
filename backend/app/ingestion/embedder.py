"""Embedding generation.

Future responsibilities:
- loading the configured EMBEDDING_MODEL (sentence-transformers / HuggingFace)
- embedding chunks for storage in the vector store
- batch API to avoid repeated model calls
"""


def embed_documents(*args, **kwargs):
    """Generate embeddings for a list of chunk texts.

    TODO (later phase): instantiate the configured embedding model and return
    a vector per chunk. Not implemented in Phase 0.
    """
    raise NotImplementedError
