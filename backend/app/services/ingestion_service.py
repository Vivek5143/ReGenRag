"""Ingestion pipeline orchestrator.

Takes an uploaded ``Document`` (status UPLOADED) and drives it through the full
processing pipeline — load, clean, chunk, embed, persist — before flipping it
to PROCESSED. Everything is logged with structured key=value lines and never
the raw document contents.

The final DB write (all chunk rows + the PROCESSED status) happens in a single
transaction, so a failure mid-persist rolls back cleanly and the document is
marked FAILED instead of being left half-ingested.
"""

from sqlalchemy import delete
from sqlalchemy.orm import Session as DbSession

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.db.models.session import utcnow
from app.ingestion import chunker, cleaner, embedder, loader
from app.services import storage

logger = get_logger(__name__)


def ingest_document(db: DbSession, document: Document) -> Document:
    """Process ``document`` end-to-end. Returns it with status PROCESSED.

    On any failure the document is marked FAILED (its file is left in place for
    inspection) and the underlying error is re-raised as
    ``DocumentProcessingError``.
    """
    document_id = document.id
    logger.info(
        "document ingestion started document_id=%s filename=%s session_id=%s",
        document_id, document.filename, document.session_id,
    )
    document.status = DocumentStatus.PROCESSING
    db.commit()
    db.refresh(document)

    try:
        pages = _load_and_clean(document)
        chunks = chunker.chunk_pages(
            pages,
            document_id=document.id,
            filename=document.filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info(
            "chunks created document_id=%s chunk_count=%s",
            document_id, len(chunks),
        )

        texts = [chunk.content for chunk in chunks]
        logger.info(
            "embedding generation started document_id=%s calls=%s",
            document_id, len(texts),
        )
        vectors = embedder.embed_texts(texts)
        logger.info(
            "embedding generation completed document_id=%s vectors=%s",
            document_id, len(vectors),
        )

        _persist_chunks(db, document, chunks, vectors)
    except Exception as exc:
        _mark_failed(db, document)
        logger.error(
            "ingestion failed document_id=%s error=%s",
            document_id, exc,
            exc_info=True,
        )
        raise DocumentProcessingError(
            f"Ingestion failed for document {document_id}: {exc}"
        ) from exc

    logger.info("document marked processed document_id=%s", document_id)
    return document


def _load_and_clean(document: Document) -> list[loader.LoadedPage]:
    """Read the PDF and clean each page, preserving page order and numbers."""
    path = storage.resolve_upload_path(
        document.storage_path, settings.upload_dir
    )
    pages = loader.load_document(path, source=document.filename)
    for page in pages:
        page.text = cleaner.clean_text(page.text)

    # A PDF with no extractable text (e.g. image-only / scanned) cannot be
    # chunked or embedded; treat it as unprocessable.
    if not any(page.text for page in pages):
        raise DocumentProcessingError(
            f"PDF '{document.filename}' contains no extractable text"
        )

    logger.info(
        "pdf extraction completed document_id=%s pages=%s",
        document.id, len(pages),
    )
    return pages


def _persist_chunks(
    db: DbSession,
    document: Document,
    chunks: list[chunker.Chunk],
    vectors: list[list[float]],
) -> None:
    """Write all chunks + embeddings and mark PROCESSED in one transaction."""
    db.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
    )
    for chunk, vector in zip(chunks, vectors):
        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_number=chunk.page_number,
                meta=chunk.metadata,
                embedding=vector,
            )
        )
    document.status = DocumentStatus.PROCESSED
    document.processed_at = utcnow()
    db.commit()
    db.refresh(document)
    logger.info("chunks persisted document_id=%s chunk_count=%s", document.id, len(chunks))


def _mark_failed(db: DbSession, document: Document) -> None:
    """Roll back any partial writes and set the document to FAILED."""
    db.rollback()
    # Re-fetch defensively: rollback expires tracked instances.
    stored = db.get(Document, document.id)
    if stored is not None:
        stored.status = DocumentStatus.FAILED
        db.commit()
        db.refresh(stored)


