# ReGenRAG — Development Guide

## Development principles

- **Clean architecture where practical** — separate API, services, and the
  future RAG layers; each package has a single responsibility.
- **Separation of concerns** — routers only handle HTTP; business logic lives
  in `services/`; RAG concerns are isolated per package.
- **Type hints everywhere** — both Python and TypeScript.
- **Small, focused modules** — no god-files.
- **No unnecessary abstractions** — introduce an interface only when there are
  real, distinct implementations behind it.
- **No duplicated configuration** — one settings source of truth
  (`app/core/config.py`) driven by environment variables.
- **No secrets in source code** — credentials only via environment / `.env`,
  which is git-ignored. `.env.example` documents the keys with empty values.
- **Meaningful names** — names describe intent, not implementation.
- **Minimal but useful comments** — comments explain *why*, not *what*.
- **No placeholder code pretending to be functional** — Phase 0 stubs raise
  `NotImplementedError` and carry explicit TODO docstrings; they never return
  fake data.
- **No unnecessary dependencies** — keep `requirements.txt` and `package.json`
  lean; add packages only when a phase needs them.

## Environment configuration

Copy `backend/.env.example` to `backend/.env` and fill in values. The settings
class in `app/core/config.py` reads it on startup. Never commit a real `.env`.

## Local setup

See the root [README](../README.md#local-setup) for full instructions.

## Testing

- Unit tests run with **no** LLM keys, **no** embedding model downloads,
  **no** database server, and **no** network access.
- Run backend tests:

  ```bash
  cd backend
  python -m pytest
  ```

- Tests live in `backend/tests/unit/` (isolated) and
  `backend/tests/integration/` (may require a running database).

## Roadmap / future phases

| Phase | Scope |
| ----- | ----- |
| **Phase 0** | Project foundation: app shell, config, DB foundation, health endpoint, placeholder modules, minimal UI, Docker. **No RAG pipeline.** ✅ |
| **Phase 1** | Database & temporary session infrastructure: PostgreSQL engine, ORM models (sessions/documents/chunks), Alembic migrations, pgvector schema, upload API, idempotent cleanup, minimal UI. **No RAG pipeline.** ✅ |
| **Phase 2** | Ingestion: PDF loading (pypdf), text cleaning, chunking, local embeddings (sentence-transformers), pgvector storage. ✅ |
| **Phase 3 (current)** | Retrieval + generation: vector search, grounded answer generation, refusal on insufficient evidence. |
| Phase 4 | Self-healing loop: LangGraph workflow, query rewriting, retrieval grading, answer criticism, retries. |
| Phase 5 | Evaluation: retrieval/answer metrics, offline evaluation harness. |
| Phase 6 | Frontend: chat UI, self-healing trace, sources/evidence view. |

### Notes for contributors

- Do not start implementing a future phase until the current phase is verified
  (see the root README's *Current project status*).
- Migrations live in `backend/alembic/` and are driven by the application's own
  settings and metadata (`alembic/env.py`). Run `alembic upgrade head` from the
  `backend/` directory.
- The `document_chunks.embedding` column (pgvector) is `Vector(384)`, matching
  the default embedding model `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
  The embedder verifies the model's real output dimension against
  `EMBEDDING_DIMENSION` at load time and refuses to run on a mismatch. Changing
  either requires a new migration.
