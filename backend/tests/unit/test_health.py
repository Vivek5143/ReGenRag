"""Tests for GET /health (liveness) and GET /ready (readiness).

/health - liveness only, does not check database
/ready  - readiness, checks database connectivity
"""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_liveness_always_ok():
    """Liveness endpoint should always return 200 if process is alive."""
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ReGenRAG"
    # liveness does NOT include database field


def test_ready_ok_when_database_reachable():
    """Readiness endpoint returns 200 when database is reachable."""
    client = TestClient(create_app())
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ReGenRAG"
    assert body["database"] == "connected"


def test_ready_reports_degraded_when_database_unreachable(monkeypatch):
    """Readiness endpoint returns 503 when database is unreachable."""
    import app.api.routes.health as health_module

    def _raise(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(health_module, "get_engine", _raise)

    client = TestClient(create_app())
    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "disconnected"