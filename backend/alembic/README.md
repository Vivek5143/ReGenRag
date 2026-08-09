# Alembic (database migrations)

Alembic is configured to use the application's SQLAlchemy metadata and
environment-driven settings, so it always migrates the same schema the app
uses.

## Upgrading

Run from the `backend/` directory (uses `DATABASE_URL` from `backend/.env`):

```bash
alembic upgrade head
alembic current
```

## Phase 1 status

- Initial migration: `0001` — creates `sessions`, `documents`, and
  `document_chunks`, enables the `vector` extension, and adds the `embedding`
  column (prepared but unused until an embedding model is selected).
- Migrations are authored by hand; `alembic revision --autogenerate` should only
  be used after the schema is reviewed.

## Adding a migration

```bash
cd backend
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```