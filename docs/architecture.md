# ReGenRAG — Planned Architecture

## Overview

ReGenRAG is a self-healing Retrieval-Augmented Generation (RAG) system for
reliable, evidence-grounded document intelligence. Users open a temporary
**session**, upload PDFs, ask questions, and receive answers grounded in the
retrieved evidence. When retrieval or generation quality is poor, the system
heals itself by rewriting the query and retrying — and it **refuses to answer**
when sufficient evidence cannot be found.

Uploaded documents are **session-scoped and ephemeral**. When a session ends
or expires, its PDFs, chunks, and embeddings are removed.

## Core data model

```text
Session
   -> Documents
        -> Chunks
             -> Embeddings
```

Everything related to uploaded documents belongs to a session, which makes
cleanup at session end a natural cascade (delete session -> delete its
documents/chunks/embeddings).

## High-level architecture

```text
+---------------------+     +-------------------------+
|  React + Vite UI     |---->|  FastAPI (REST)         |
|  Upload / Chat /     |     |  /health /sessions ...  |
|  Trace / Evidence    |     +------------+------------+
+---------------------+                  |
                                         v
                              +--------------------------+
                              |  Application services     |
                              |  session_service          |
                              +------------+-------------+
                                           |
        +----------------------+-----------+------------+--------------------+
        v                      v                        v                    v
+---------------+     +----------------+       +---------------+     +----------------+
|  Ingestion    |     |  Retrieval     |       |  Generation   |     |  Evaluation    |
|  load/chunk/  |     |  vector store  |       |  LLM provider |     |  metrics /     |
|  embed        |     |  retriever     |       |  grounded     |     |  evaluator     |
+---------------+     +----------------+       |  answer       |     +----------------+
                          |                    +---------------+            ^
                          v                                                  |
                 +-----------------+                                        |
                 |  PostgreSQL +    |                                        |
                 |  pgvector        |----------------------------------------+
                 |  (sessions,      |
                 |   chunks,        |
                 |   embeddings)    |
                 +-----------------+
                          ^
                          |
              +-----------+-----------+
              |  LangGraph workflow  |
              |  rewrite -> retrieve |
              |  -> grade ->         |
              |  generate ->         |
              |  critique (retry)    |
              +----------------------+
```

## The self-healing loop (planned)

The core differentiator of ReGenRAG is the self-healing LangGraph loop:

1. **Rewrite query** — rephrase a weak query before retrieval.
2. **Retrieve** — fetch top-k evidence chunks for the session (vector search).
3. **Grade retrieval** — is the retrieved evidence sufficient for an answer?
4. **Generate** — produce an answer grounded strictly in the evidence.
5. **Critique answer** — is the answer faithful to the evidence?
6. **Retry** — if retrieval or grounding is poor, rewrite and retry up to
   `MAX_RETRIES`.
7. **Refuse** — if evidence is still insufficient, decline to answer rather
   than hallucinate.

The workflow state contract lives in `backend/app/graph/state.py`.

## Layer responsibilities

| Layer              | Path                            | Responsibility                          |
| ------------------ | ------------------------------- | --------------------------------------- |
| API                | `app/api/routes/`               | HTTP endpoints, request/response models |
| Core / config      | `app/core/`                     | Environment-driven settings             |
| Database           | `app/db/`                       | SQLAlchemy models + engine foundation   |
| Services           | `app/services/`                 | Business logic (session lifecycle)      |
| Ingestion          | `app/ingestion/`                | PDF load, chunk, embed (future)         |
| Retrieval          | `app/retrieval/`                | Vector store + retriever (future)       |
| Generation         | `app/generation/`               | LLM-grounded answers (future)           |
| Graph              | `app/graph/`                    | LangGraph state + nodes + workflow      |
| Evaluation         | `app/evaluation/`               | Retrieval/answer metrics (future)       |
| Frontend           | `frontend/src/`                 | React UI (upload, chat, trace, sources) |

## Frontend flow (planned)

```text
Document Upload  ->  Document List  ->  Chat Interface
                                          |
                                          v
                                  Self-Healing Trace
                                          |
                                          v
                                   Sources / Evidence
```

## Persistence & cleanup

- PostgreSQL with pgvector stores sessions, chunks, and embeddings.
- Uploaded files live under `data/uploads/`; processed artifacts under
  `data/processed/`.
- A background cleanup job (or lazy expiry check) deletes expired sessions and
  their artifacts (PDFs, chunks, embeddings) in a later phase.

## Tech stack

- **Backend:** Python 3.11+, FastAPI, Pydantic / pydantic-settings, SQLAlchemy,
  Alembic
- **AI/RAG:** LangChain, LangGraph, Sentence Transformers / HuggingFace
  embeddings, configurable LLM provider (env-driven), PostgreSQL + pgvector
- **Frontend:** React, TypeScript, Vite
- **Testing:** Pytest
- **Infrastructure:** Docker, Docker Compose
