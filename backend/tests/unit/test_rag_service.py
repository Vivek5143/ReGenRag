"""Tests for the baseline RAG service orchestration.

Retrieval and generation are faked so the service's own responsibilities —
validation, context building, and source projection — are tested in isolation.
"""

import uuid

import pytest

from app.core.exceptions import NoProcessedDocumentError, SessionNotFoundError
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.retrieval.vector_store import RetrievedChunk
from app.services import rag_service, session_service


def _binding_error_session(db_session):
    """A random id that is guaranteed not to exist."""
    return uuid.uuid4()


def _with_processed_document(db_session):
    session = session_service.create_session(db_session)
    db_session.add(
        Document(
            id=uuid.uuid4(),
            session_id=session.id,
            filename="report.pdf",
            stored_filename="x.pdf",
            content_type="application/pdf",
            file_size=5,
            storage_path=f"{session.id}/report.pdf",
            status=DocumentStatus.PROCESSED,
        )
    )
    db_session.commit()
    return session


def _fake_chunk(page, text, index):
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=index,
        page_number=page,
        content=text,
        score=0.8,
    )


def test_invalid_session_raises_not_found(db_session):
    with pytest.raises(SessionNotFoundError):
        rag_service.answer_question(db_session, _binding_error_session(db_session), "hi")


def test_no_processed_document_raises(db_session):
    session = session_service.create_session(db_session)  # no documents at all

    with pytest.raises(NoProcessedDocumentError):
        rag_service.answer_question(db_session, session.id, "hi")


def test_blank_question_rejected(db_session):
    session = session_service.create_session(db_session)

    with pytest.raises(ValueError):
        rag_service.answer_question(db_session, session.id, "   ")


def test_happy_path_returns_answer_and_sources(db_session, monkeypatch):
    session = _with_processed_document(db_session)
    chunks = [_fake_chunk(4, "alpha", 0), _fake_chunk(7, "beta", 1)]

    def fake_retrieve(db, session_id, question, top_k=None):
        assert session_id == session.id
        return chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        rag_service.generator, "generate_answer", lambda q, c: "Generated answer."
    )

    result = rag_service.answer_question(db_session, session.id, "What is this?")

    assert result.answer == "Generated answer."
    assert len(result.sources) == 2
    assert result.sources[0].page_number == 4
    assert result.sources[0].chunk_index == 0
    assert result.sources[1].page_number == 7
    assert result.sources[1].chunk_index == 1
    assert result.sources[0].score == 0.8


def test_empty_retrieval_yields_empty_sources(db_session, monkeypatch):
    session = _with_processed_document(db_session)

    monkeypatch.setattr(rag_service.retriever, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(
        rag_service.generator, "generate_answer", lambda q, c: "No info available."
    )

    result = rag_service.answer_question(db_session, session.id, "Anything?")

    assert result.answer == "No info available."
    assert result.sources == []


def test_retrieval_failure_wrapped_as_retrieval_error(db_session, monkeypatch):
    from app.core.exceptions import RetrievalError

    session = _with_processed_document(db_session)

    def broken_retrieve(*args, **kwargs):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(rag_service.retriever, "retrieve", broken_retrieve)

    with pytest.raises(RetrievalError):
        rag_service.answer_question(db_session, session.id, "hi")
