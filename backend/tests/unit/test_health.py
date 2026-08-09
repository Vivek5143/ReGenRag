"""Tests for GET /health.

Requires a reachable database to assert "connected"; no external services.
"""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok_when_database_reachable():
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ReGenRAG"
    assert body["database"] == "connected"


def test_health_reports_degraded_when_database_unreachable(monkeypatch):
    import app.api.routes.health as health_module

    def _raise(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(health_module, "get_engine", _raise)

    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "disconnected"