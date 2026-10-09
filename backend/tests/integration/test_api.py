"""End-to-end API tests for the session + document lifecycle.

Run against the isolated test database via the client fixture (get_db is
overridden to the test DB; uploads land in a temp dir). A fake embedding model
is installed globally in conftest, so ingestion runs end-to-end with no
network, no model downloads, and no LLM.
"""

from tests.factories import pdf_bytes

# A real, parseable text PDF. Multiple pages so page handling is exercised.
REAL_PDF = pdf_bytes(
    "First page of the report body.",
    "Second page with more evidence.",
)


def _upload_pdf(name="report.pdf", content=REAL_PDF, content_type="application/pdf"):
    return {"file": (name, content, content_type)}


def test_create_session_endpoint(client):
    response = client.post("/api/v1/sessions")

    assert response.status_code == 201
    body = response.json()
    assert "session_id" in body
    assert body["status"] == "ACTIVE"
    assert "expires_at" in body


def test_get_session_endpoint(client):
    created = client.post("/api/v1/sessions").json()

    response = client.get(f"/api/v1/sessions/{created['session_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == created["session_id"]
    assert body["status"] == "ACTIVE"


def test_get_missing_session_returns_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/v1/sessions/{missing}").status_code == 404


def test_upload_document_endpoint(client):
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/documents", files=_upload_pdf()
    )

    assert response.status_code == 201
    body = response.json()
    assert body["session_id"] == session_id
    assert body["filename"] == "report.pdf"
    assert body["status"] == "PROCESSED"
    assert body["chunk_count"] >= 1
    # Internal physical storage path must never be exposed to the client.
    assert "storage_path" not in body


def test_upload_malformed_pdf_marks_failed(client):
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/documents",
        files=_upload_pdf(content=b"%PDF-1.4 fake bytes"),
    )

    # Ingestion failed: the endpoint reports an error and the document is kept
    # in FAILED state so the client can inspect it.
    assert response.status_code == 500
    docs = client.get(f"/api/v1/sessions/{session_id}/documents").json()
    assert len(docs) == 1
    assert docs[0]["status"] == "FAILED"


def test_list_documents_endpoint(client):
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.post(f"/api/v1/sessions/{session_id}/documents", files=_upload_pdf())

    response = client.get(f"/api/v1/sessions/{session_id}/documents")

    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "report.pdf"
    assert docs[0]["status"] == "PROCESSED"
    assert docs[0]["chunk_count"] >= 1
    assert "storage_path" not in docs[0]


def test_upload_rejects_non_pdf(client):
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["error"]["detail"]


def test_upload_rejects_oversized(client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_file_size_mb", 0.001)  # ~1 KB
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/documents",
        files={"file": ("big.pdf", b"x" * 4096, "application/pdf")},
    )

    assert response.status_code == 400


def test_upload_to_closed_session_rejected(client):
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.delete(f"/api/v1/sessions/{session_id}")

    response = client.post(
        f"/api/v1/sessions/{session_id}/documents", files=_upload_pdf()
    )

    assert response.status_code == 409


def test_close_session_endpoint_cleans_up(client):
    created = client.post("/api/v1/sessions").json()
    session_id = created["session_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/documents",
        files=_upload_pdf(),
    )

    response = client.delete(f"/api/v1/sessions/{session_id}")

    assert response.status_code == 204
    # Session is now CLOSED and its documents are gone.
    get_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.json()["status"] == "CLOSED"
    docs = client.get(f"/api/v1/sessions/{session_id}/documents")
    assert docs.json() == []