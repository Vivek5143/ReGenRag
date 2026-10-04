"""End-to-end API tests for the Phase 3 query endpoint.

Run against the isolated test database via the server fixture (get_db is
overridden to the test DB). A fake embedding model is installed globally in
conftest and a fake LLM is installed here, so no network and no provider
credentials are ever used.
"""

import uuid

import pytest

from app.core.config import settings
from app.core.exceptions import LLMError
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.generation import llm as llm_module
from app.services import rag_service

# NOTE: Phase 4 insufficient-evidence gating uses deterministic
# similarity scores from pgvector. For these API tests we patch the
# retriever embedding so cosine similarity is finite and predictable.

DIM = 384


def _vec(leading: float) -> list[float]:
    out = [0.0] * DIM
    out[0] = float(leading)
    return out


class FakeLLM:
    """Records the last prompt; returns a canned answer or raises."""

    def __init__(self, response="This document appears to cover the topic.", error=None):
        self.response = response
        self.error = error
        self.last_prompt = None

    def complete(self, prompt, *, max_tokens=512):
        self.last_prompt = prompt
        if self.error:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch):
    """Install a fake LLM so no real provider is ever called."""
    monkeypatch.setattr(llm_module, "_client", FakeLLM())


def _session_id(client):
    return client.post("/api/v1/sessions").json()["session_id"]


def _seed_processed_document(db_session, session_id):
    """Insert a PROCESSED document with two searchable chunks."""
    doc = Document(
        id=uuid.uuid4(),
        session_id=session_id,
        filename="report.pdf",
        stored_filename="x.pdf",
        content_type="application/pdf",
        file_size=5,
        storage_path=f"{session_id}/report.pdf",
        status=DocumentStatus.PROCESSED,
    )
    db_session.add(doc)
    db_session.commit()
    chunk_a = DocumentChunk(
        id=uuid.uuid4(), document_id=doc.id, chunk_index=0,
        content="The report analyzes quarterly revenue.", page_number=1,
        embedding=_vec(1.0),
    )
    chunk_b = DocumentChunk(
        id=uuid.uuid4(), document_id=doc.id, chunk_index=1,
        content="Outlook for the next fiscal year.", page_number=3,
        embedding=_vec(0.9),
    )
    db_session.add_all([chunk_a, chunk_b])
    db_session.commit()
    return doc


def test_query_successful(client, db_session, monkeypatch):
    session_id = _session_id(client)
    _seed_processed_document(db_session, session_id)

    # Ensure deterministic, non-zero query embeddings so pgvector cosine
    # similarity is finite and the Phase 4 similarity sufficiency gate passes.
    monkeypatch.setattr(
        rag_service.retriever,
        "embed_query",
        lambda q: _vec(1.0),
    )

    response = client.post(
        f"/api/v1/sessions/{session_id}/query",
        json={"question": "What is this document about?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "This document appears to cover the topic."
    assert len(body["sources"]) >= 1
    for source in body["sources"]:
        assert "page_number" in source
        assert "chunk_index" in source
        assert "document_id" in source
        assert "chunk_id" in source


def test_query_invalid_session(client):
    missing = "00000000-0000-0000-0000-000000000000"

    response = client.post(
        f"/api/v1/sessions/{missing}/query", json={"question": "hi"}
    )

    assert response.status_code == 404


def test_query_no_processed_document(client, db_session):
    session_id = _session_id(client)  # session exists, but no documents

    response = client.post(
        f"/api/v1/sessions/{session_id}/query", json={"question": "hi"}
    )

    assert response.status_code == 404
    assert "processed" in response.json()["detail"].lower()


def test_query_empty_question_rejected(client):
    session_id = _session_id(client)

    response = client.post(
        f"/api/v1/sessions/{session_id}/query", json={"question": ""}
    )

    assert response.status_code == 422


def test_query_blank_question_rejected(client):
    session_id = _session_id(client)

    response = client.post(
        f"/api/v1/sessions/{session_id}/query", json={"question": "   "}
    )

    assert response.status_code == 422


def test_query_no_retrieved_chunks_is_graceful(client, db_session, monkeypatch):
    session_id = _session_id(client)
    _seed_processed_document(db_session, session_id)
    monkeypatch.setattr(rag_service.retriever, "retrieve", lambda *a, **k: [])

    response = client.post(
        f"/api/v1/sessions/{session_id}/query", json={"question": "Anything?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert isinstance(body["answer"], str) and body["answer"]

    # Phase 4: the API exposes failure_category on the no-results path.
    assert body["failure_category"] == "NO_RESULTS"
    assert body["rewritten_query"] is None
    assert body["retrieval_grading"] is not None
    assert body["retrieval_grading"]["sufficient"] is False


def test_query_llm_failure_returns_502(client, db_session, monkeypatch):
    session_id = _session_id(client)
    _seed_processed_document(db_session, session_id)
    monkeypatch.setattr(
        llm_module, "_client", FakeLLM(error=LLMError("provider down"))
    )

    response = client.post(
        f"/api/v1/sessions/{session_id}/query", json={"question": "hi"}
    )

    assert response.status_code == 502


def test_query_llm_not_configured_returns_503(client, db_session, monkeypatch):
    session_id = _session_id(client)
    _seed_processed_document(db_session, session_id)
    monkeypatch.setattr(llm_module, "_client", None)
    monkeypatch.setattr(settings, "llm_provider", "")

    response = client.post(
        f"/api/v1/sessions/{session_id}/query", json={"question": "hi"}
    )

    assert response.status_code == 503


def test_query_response_includes_rewritten_query(client, db_session, monkeypatch):
    """The API exposes rewritten_query, mapped from RagAnswer.rewritten_query."""

    session_id = _session_id(client)
    _seed_processed_document(db_session, session_id)

    # Deterministic embedding so the initial retrieval succeeds and generates.
    monkeypatch.setattr(
        rag_service.retriever, "embed_query", lambda q: _vec(1.0)
    )

    captured = {}

    def fake_answer_question(db, sid, question):
        captured["called"] = True
        return rag_service.RagAnswer(
            answer="ok",
            sources=[],
            rewritten_query="rewritten version of the question",
        )

    monkeypatch.setattr(
        rag_service, "answer_question", fake_answer_question
    )

    response = client.post(
        f"/api/v1/sessions/{session_id}/query",
        json={"question": "What is this?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert captured.get("called") is True
    assert "rewritten_query" in body
    assert body["rewritten_query"] == "rewritten version of the question"
