"""Tests for vector-store similarity search and the retriever.

These exercise the real pgvector path against the test database (vector
extension created in conftest). Embeddings are hand-crafted nonzero vectors so
similarity ordering is deterministic; the global fake embedding model is used
only to test that a query is embedded.
"""

import uuid

import pytest

from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.retrieval import retriever, vector_store
from app.services import session_service

DIM = 384


def _vec(*leading: float) -> list[float]:
    """A 384-dim vector with the given leading values and trailing zeros."""
    out = [0.0] * DIM
    for i, value in enumerate(leading):
        out[i] = float(value)
    return out


def _processed_document(db, session_id, filename="report.pdf"):
    doc = Document(
        id=uuid.uuid4(),
        session_id=session_id,
        filename=filename,
        stored_filename=f"{uuid.uuid4()}.pdf",
        content_type="application/pdf",
        file_size=5,
        storage_path=f"{session_id}/{filename}",
        status=DocumentStatus.PROCESSED,
    )
    db.add(doc)
    db.commit()
    return doc


def _add_chunk(db, document, index, *, page, content, vector):
    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=document.id,
        chunk_index=index,
        content=content,
        page_number=page,
        embedding=vector,
    )
    db.add(chunk)
    db.commit()
    return chunk


@pytest.fixture()
def session_and_doc(db_session):
    session = session_service.create_session(db_session)
    doc = _processed_document(db_session, session.id)
    return session, doc


# --- vector_store.similarity_search -----------------------------------------


def test_returns_top_k_most_similar(db_session, session_and_doc):
    session, doc = session_and_doc
    c1 = _add_chunk(
        db_session, doc, 0, page=1, content="alpha", vector=_vec(1.0)
    )
    c2 = _add_chunk(
        db_session, doc, 1, page=2, content="beta", vector=_vec(0.5, 0.5)
    )
    _add_chunk(db_session, doc, 2, page=3, content="gamma", vector=_vec(0.0, 1.0))

    # Query points most strongly toward [1, 0, 0, ...]: alpha, then beta, then gamma.
    results = vector_store.similarity_search(
        db_session, _vec(1.0), session_id=session.id, top_k=2
    )

    assert [r.chunk_id for r in results] == [c1.id, c2.id]
    assert results[0].score >= results[1].score


def test_orders_by_similarity_descending(db_session, session_and_doc):
    session, doc = session_and_doc
    near = _add_chunk(db_session, doc, 0, page=1, content="near", vector=_vec(0.99, 0.01))
    far = _add_chunk(db_session, doc, 1, page=2, content="far", vector=_vec(0.0, 1.0))
    mid = _add_chunk(db_session, doc, 2, page=3, content="mid", vector=_vec(0.5, 0.5))

    results = vector_store.similarity_search(
        db_session, _vec(1.0), session_id=session.id, top_k=10
    )

    assert [r.chunk_id for r in results] == [near.id, mid.id, far.id]
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
    assert results[0].score > results[1].score > results[2].score


def test_score_is_a_real_float_for_matching_query(db_session, session_and_doc):
    # Regression: the query embedding must be passed as a plain list so the
    # cosine-distance operator binds it and returns a float; wrapping it in a
    # Vector type instance previously made float() fail on the score.
    session, doc = session_and_doc
    query = _vec(1.0)
    _add_chunk(db_session, doc, 0, page=1, content="match", vector=query)

    (result,) = vector_store.similarity_search(
        db_session, query, session_id=session.id, top_k=1
    )

    assert isinstance(result.score, float)
    assert result.score == pytest.approx(1.0, abs=1e-3)


def test_scopes_results_to_session(db_session):
    session_a = session_service.create_session(db_session)
    session_b = session_service.create_session(db_session)
    doc_a = _processed_document(db_session, session_a.id)
    doc_b = _processed_document(db_session, session_b.id)
    chunk_a = _add_chunk(db_session, doc_a, 0, page=1, content="only in A", vector=_vec(1.0))
    chunk_b = _add_chunk(db_session, doc_b, 0, page=1, content="only in B", vector=_vec(1.0))

    results = vector_store.similarity_search(
        db_session, _vec(1.0), session_id=session_a.id, top_k=10
    )

    assert [r.chunk_id for r in results] == [chunk_a.id]
    assert chunk_b.id not in [r.chunk_id for r in results]


def test_preserves_source_metadata(db_session, session_and_doc):
    session, doc = session_and_doc
    chunk = _add_chunk(
        db_session, doc, 7, page=4, content="some text here", vector=_vec(1.0)
    )

    (result,) = vector_store.similarity_search(
        db_session, _vec(1.0), session_id=session.id, top_k=1
    )

    assert result.document_id == doc.id
    assert result.chunk_id == chunk.id
    assert result.chunk_index == 7
    assert result.page_number == 4
    assert result.content == "some text here"
    assert result.score is not None


def test_empty_when_session_has_no_chunks(db_session, session_and_doc):
    session, _ = session_and_doc

    results = vector_store.similarity_search(
        db_session, _vec(1.0), session_id=session.id, top_k=5
    )

    assert results == []


def test_ignores_chunks_without_embedding(db_session, session_and_doc):
    session, doc = session_and_doc
    _add_chunk(db_session, doc, 0, page=1, content="no embedding", vector=None)

    results = vector_store.similarity_search(
        db_session, _vec(1.0), session_id=session.id, top_k=5
    )

    assert results == []


# --- retriever ---------------------------------------------------------------


def test_embed_query_returns_384_dim_vector():
    vector = retriever.embed_query("what is this about")

    assert len(vector) == DIM


def test_retrieve_embeds_query_and_delegates_to_store(db_session, session_and_doc, monkeypatch):
    session, _ = session_and_doc
    captured = {}

    def fake_embed(question):
        captured["question"] = question
        return _vec(1.0)

    def fake_search(db, query_embedding, *, session_id, top_k):
        captured["embedding"] = query_embedding
        captured["session_id"] = session_id
        captured["top_k"] = top_k
        return [vector_store.RetrievedChunk(
            chunk_id=uuid.uuid4(), document_id=uuid.uuid4(),
            chunk_index=0, page_number=1, content="hit", score=0.9,
        )]

    monkeypatch.setattr(retriever, "embed_query", fake_embed)
    monkeypatch.setattr(retriever.vector_store, "similarity_search", fake_search)

    results = retriever.retrieve(db_session, session.id, "a question", top_k=3)

    assert len(results) == 1
    assert captured["question"] == "a question"
    assert captured["embedding"] == _vec(1.0)
    assert captured["session_id"] == session.id
    assert captured["top_k"] == 3
