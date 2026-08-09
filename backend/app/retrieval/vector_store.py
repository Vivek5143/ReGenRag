"""Vector store.

Future responsibilities:
- PostgreSQL + pgvector storage for chunk embeddings
- scoping all stored vectors to a session
- deletion of a session's vectors on session end/expiry
- similarity search queries
"""


class VectorStore:
    """Session-scoped vector store boundary.

    TODO (later phase): persist/query embeddings via pgvector. Not implemented
    in Phase 0.
    """

    def add(self, *args, **kwargs):
        raise NotImplementedError

    def search(self, *args, **kwargs):
        raise NotImplementedError

    def delete_for_session(self, *args, **kwargs):
        raise NotImplementedError
