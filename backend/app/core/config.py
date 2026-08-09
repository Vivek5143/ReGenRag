"""Application settings, loaded from environment variables.

All configuration is environment-driven. No secrets or provider-specific
credentials are hardcoded here. Values are read from a ``.env`` file at the
backend root (or the process environment) via pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Paths in this file are resolved relative to the repository root. This module
# lives at <root>/backend/app/core/config.py, so:
#   parents[0] = app/core, parents[1] = app, parents[2] = backend,
#   parents[3] = repository root.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """Typed application settings with sane defaults for development."""

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Ignore empty values so defaults apply when .env leaves a key blank.
        env_ignore_empty=True,
    )

    # --- Application -------------------------------------------------------
    app_name: str = "ReGenRAG"
    app_env: str = "development"

    # --- Database ----------------------------------------------------------
    # PostgreSQL + pgvector URL, e.g.
    # postgresql+psycopg2://user:pass@localhost:5432/regenrag
    database_url: str = ""
    # Optional isolated database used by the test suite. If left empty, tests
    # derive one from DATABASE_URL (e.g. "regenrag" -> "regenrag_test").
    test_database_url: str = ""

    # --- LLM provider (configurable, never hardcoded credentials) ----------
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""

    # --- Embeddings ---------------------------------------------------------
    embedding_model: str = ""
    # Embedding dimension for the pgvector column. Chosen to match the common
    # sentence-transformers size (384). It must be aligned with whichever
    # embedding model is selected in a later phase; changing it requires a new
    # migration. No embeddings are generated or stored in Phase 1.
    embedding_dimension: int = 384

    # --- Storage ------------------------------------------------------------
    upload_dir: Path = _PROJECT_ROOT / "data" / "uploads"
    processed_dir: Path = _PROJECT_ROOT / "data" / "processed"

    # --- RAG / session tuning ------------------------------------------------
    max_file_size_mb: int = 20
    session_ttl_minutes: int = 60
    max_retries: int = 3
    rag_top_k: int = 5

    @property
    def max_upload_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @model_validator(mode="after")
    def _resolve_relative_dirs(self) -> "Settings":
        """Resolve relative storage dirs against the repository root."""
        if not self.upload_dir.is_absolute():
            self.upload_dir = (_PROJECT_ROOT / self.upload_dir).resolve()
        if not self.processed_dir.is_absolute():
            self.processed_dir = (_PROJECT_ROOT / self.processed_dir).resolve()
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached so that a single process reads its configuration once and reuses
    it across request handlers.
    """
    return Settings()


settings = get_settings()