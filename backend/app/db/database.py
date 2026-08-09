"""SQLAlchemy engine and session foundation.

PostgreSQL is the intended database. Configuration is entirely environment
driven; no credentials appear in source code.

A single engine is created lazily and shared across the application. Tests
build an isolated engine via make_engine() against a dedicated test database.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def make_engine(database_url: str | None = None) -> Engine:
    """Build a SQLAlchemy engine with pooling appropriate for local development."""
    url = database_url or settings.database_url
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Copy backend/.env.example to "
            "backend/.env and set DATABASE_URL."
        )
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


_engine: Engine | None = None
_session_local: sessionmaker | None = None


def get_engine() -> Engine:
    """Return the shared application engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_local() -> sessionmaker:
    """Return a Session factory bound to the application engine."""
    global _session_local
    if _session_local is None:
        _session_local = sessionmaker(
            bind=get_engine(), class_=Session, expire_on_commit=False
        )
    return _session_local


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session (closed on exit)."""
    db = get_session_local()()
    try:
        yield db
    finally:
        db.close()