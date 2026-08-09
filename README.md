# ReGenRAG

**Self-healing Retrieval-Augmented Generation for reliable, evidence-grounded document intelligence.**

> **Current version: Project Foundation / Phase 0**
>
> This repository currently establishes the **project foundation only**. The RAG
> pipeline (PDF parsing, chunking, embeddings, vector search, LLM calls, the
> LangGraph self-healing loop, and evaluation) has **not yet been implemented**.
> See [Current project status](#current-project-status).

## What is ReGenRAG?

ReGenRAG lets users open a temporary document session, upload PDFs, and ask
questions about them. The system retrieves evidence from the uploaded documents
and generates answers grounded strictly in that evidence. Its defining feature
is **self-healing**: when retrieval or generation quality is poor, it rewrites
the query and retries — and it **refuses to answer** when sufficient evidence
cannot be found.

Uploaded documents are **session-scoped and ephemeral**: when a session ends or
expires, the uploaded PDFs, extracted chunks, and associated embeddings are
removed.

## Problem statement

Vanilla RAG systems hallucinate when retrieval is weak, return answers not
supported by the sources, and silently answer questions the document set cannot
answer. ReGenRAG addresses this by:

1. **Evaluating retrieval quality** before generating an answer.
2. **Rewriting queries** when retrieval is poor.
3. **Critiquing answers** for grounding in the retrieved evidence.
4. **Retrying** failed retrieval/generation attempts within limits.
5. **Refusing to answer** when sufficient evidence cannot be found — instead of
   guessing.

## Planned architecture

```text
React + Vite UI
   |  (REST)
   v
FastAPI          ->  services (session lifecycle)
   |                    |
   |                    +-- Ingestion  (load -> chunk -> embed)   [future]
   |                    +-- Retrieval  (vector store, retriever)  [future]
   |                    +-- Generation (grounded LLM answers)     [future]
   |                    +-- Evaluation (retrieval/answer metrics) [future]
   +-- PostgreSQL + pgvector (sessions, chunks, embeddings)
   +-- LangGraph self-healing workflow (rewrite -> retrieve -> grade
       -> generate -> critique, with retries)                     [future]
```

See [docs/architecture.md](docs/architecture.md) for the full architecture.

## Technology stack

| Area            | Technology                                             |
| --------------- | ------------------------------------------------------ |
| Backend         | Python 3.11+, FastAPI, Pydantic / pydantic-settings, SQLAlchemy, Alembic |
| AI / RAG        | LangChain, LangGraph, Sentence Transformers / HuggingFace embeddings, configurable LLM provider, PostgreSQL + pgvector |
| Frontend        | React, TypeScript, Vite                                |
| Testing         | Pytest                                                 |
| Infrastructure  | Docker, Docker Compose                                 |

## Repository structure

```text
regenrag/
├── backend/            # FastAPI application
│   ├── app/
│   │   ├── main.py         # app factory + router wiring
│   │   ├── api/routes/     # HTTP endpoints (health)
│   │   ├── core/           # environment-driven settings
│   │   ├── db/             # SQLAlchemy engine + Session model
│   │   ├── ingestion/      # [placeholder] load/chunk/embed
│   │   ├── retrieval/      # [placeholder] vector store/retriever
│   │   ├── generation/     # [placeholder] grounded answers
│   │   ├── graph/          # [placeholder] LangGraph state/nodes/workflow
│   │   ├── evaluation/     # [placeholder] metrics/evaluator
│   │   └── services/       # session lifecycle service
│   ├── tests/              # pytest suite (unit + integration)
│   ├── alembic/            # migration tooling (adopted later)
│   ├── requirements.txt
│   └── .env.example
├── frontend/           # React + TypeScript + Vite
│   └── src/                # App shell + health/status placeholder
├── data/
│   ├── uploads/            # session-scoped uploaded PDFs (git-ignored)
│   ├── processed/          # extracted artifacts (git-ignored)
│   └── evaluation/         # evaluation results (git-ignored)
├── docs/
│   ├── architecture.md
│   └── development.md
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Development roadmap

| Phase | Scope |
| ----- | ----- |
| **Phase 0** | **Foundation** — app shell, config, DB foundation, health endpoint, placeholder modules, minimal UI, Docker. ✅ |
| **Phase 1** | **Database & session infrastructure** — PostgreSQL, ORM models (sessions/documents/chunks), Alembic migrations, pgvector schema, upload API, cleanup lifecycle, minimal UI. ✅ **← you are here** |
| Phase 2 | Ingestion: PDF loading, chunking, embeddings, pgvector storage |
| Phase 3 | Retrieval + generation: vector search, grounded answers, refusal |
| Phase 4 | Self-healing loop: LangGraph workflow, rewriting, grading, criticism, retries |
| Phase 5 | Evaluation: retrieval/answer metrics, offline evaluation harness |
| Phase 6 | Full frontend: upload/list, chat, self-healing trace, sources/evidence |

## Local setup

### Prerequisites

- Python 3.11+ (3.10 works for the current foundation)
- Node.js 18+
- PostgreSQL (16+) with the pgvector extension installed
- (Optional) Docker + Docker Compose for the full stack

### Backend

The backend requires a PostgreSQL database. Set it up once, then configure env.

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then set DATABASE_URL (and optionally TEST_DATABASE_URL)
alembic upgrade head        # create the schema (sessions, documents, chunks)

uvicorn app.main:app --reload --port 8000
# -> http://localhost:8000/health  => {"status":"ok","service":"ReGenRAG","database":"connected"}
```

API endpoints (Phase 1):

```text
POST   /api/v1/sessions                        create a session
GET    /api/v1/sessions/{id}                    session metadata
DELETE /api/v1/sessions/{id}                   close a session + cleanup
POST   /api/v1/sessions/{id}/documents         upload a PDF
GET    /api/v1/sessions/{id}/documents         list session documents
```

Run tests (require PostgreSQL; no LLM keys, no embedding downloads, no network):

```bash
cd backend
python -m pytest
```

Tests use an isolated test database derived from `DATABASE_URL` (e.g.
`regenrag` → `regenrag_test`), which is created automatically if missing.
Override with `TEST_DATABASE_URL` in `.env`.

### Frontend

```bash
cd frontend
npm install
npm run dev
# -> http://localhost:5173
```

`npm run build` produces a production build in `frontend/dist/`.

### Full stack with Docker

```bash
docker compose up --build
# backend  -> http://localhost:8000
# frontend -> http://localhost:5173
# postgres -> localhost:5432 (user/pass/db: regenrag)
```

## Current project status

- **Phase 1 — Database & temporary document session infrastructure.** ✅
- The following are in place:
  - PostgreSQL connection via SQLAlchemy (pooled, lazy engine) and a
    `get_db` dependency.
  - ORM models: `Session`, `Document`, `DocumentChunk` with UUID ids, UTC
    timestamps, status enums, and cascading relationships
    (`Session 1─* Document 1─* DocumentChunk`).
  - A pgvector `embedding` column prepared on `document_chunks` (configurable
    dimension via `EMBEDDING_DIMENSION`); **no embeddings are generated yet**.
  - Alembic migrations — initial revision `0001` creates the three tables and
    enables the `vector` extension.
  - Session lifecycle service: create / get / touch (rolling expiry) / close /
    expire — with TTL from `SESSION_TTL_MINUTES`.
  - Document upload service with safe, server-generated, session-scoped
    filenames (`<session>/<document>.pdf`), PDF + MIME + size validation, and
    path-traversal protection.
  - Cleanup service (idempotent) that removes documents, chunks, and uploaded
    files for expired/closed sessions. No scheduler yet — callable on demand.
  - API endpoints for sessions and document upload under `/api/v1`, returning
    structured Pydantic models that never expose internal storage paths.
  - `GET /health` now reports database connectivity (200/`connected` or
    503/`disconnected`) without leaking credentials.
  - A minimal React + TypeScript + Vite page to start a session, upload PDFs,
    and list uploaded documents.
  - A test suite that runs against an isolated PostgreSQL test database with
    **no** LLM keys, **no** embedding downloads, and **no** network access.
  - Docker Compose with frontend, backend, and a pgvector-ready PostgreSQL.
- The **RAG pipeline is intentionally not implemented yet**: no PDF parsing,
  no chunking, no embedding generation, no vector search, no LLM calls, no
  LangGraph workflow, no query rewriting, no retrieval grading, no answer
  criticism, no evaluation metrics.

## License

See [LICENSE](LICENSE).
