"""Pytest fixtures.

Tests run against an isolated PostgreSQL test database (derived from
DATABASE_URL unless TEST_DATABASE_URL is set). They require PostgreSQL but no
LLM keys, no embedding downloads, and no network access.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.database import Base


def _resolve_test_url() -> str:
    """Return the test database URL, creating the database name if needed."""
    if settings.test_database_url:
        return settings.test_database_url
    if not settings.database_url:
        pytest.fail(
            "DATABASE_URL is not configured. Copy backend/.env.example to "
            "backend/.env and set DATABASE_URL."
        )
    url = make_url(settings.database_url)
    db_name = url.database or "regenrag"
    if not db_name.endswith("_test"):
        db_name = f"{db_name}_test"
    return url.set(database=db_name).render_as_string(hide_password=False)


def _ensure_database_exists(database_url: str) -> None:
    """Create the test database if it does not yet exist (connects as admin)."""
    url = make_url(database_url)
    admin_url = url.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": url.database},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    finally:
        admin_engine.dispose()


@pytest.fixture(scope="session")
def engine():
    """Session-scoped engine on an isolated test database with schema applied."""
    test_url = _resolve_test_url()
    _ensure_database_exists(test_url)

    eng = create_engine(test_url, pool_pre_ping=True)
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(eng)

    yield eng

    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def db_session(engine):
    """A writable session against the test database (rows cleaned per test)."""
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)
    session = TestSession()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _clean_tables(engine):
    """Reset all rows after each test for isolation."""
    yield
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture()
def client(engine, tmp_path, monkeypatch):
    """TestClient with get_db overridden to the test DB and storage in tmp."""
    from fastapi.testclient import TestClient

    from app.db.database import get_db
    from app.main import create_app

    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)

    app = create_app()

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()