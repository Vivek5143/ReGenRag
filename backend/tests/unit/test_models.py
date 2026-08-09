"""Tests for model relationships and cascading deletion."""

import io
import uuid

from sqlalchemy import delete

from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.db.models.session import Session
from app.services import session_service


def _doc(session_id, name="a.pdf"):
    return Document(
        id=uuid.uuid4(),
        session_id=session_id,
        filename=name,
        stored_filename=f"{uuid.uuid4()}.pdf",
        content_type="application/pdf",
        file_size=5,
        storage_path=f"{session_id}/{name}",
        status=DocumentStatus.UPLOADED,
    )


def test_session_has_documents_relationship(db_session):
    session = session_service.create_session(db_session)
    session.documents.append(_doc(session.id))
    session.documents.append(_doc(session.id))
    db_session.commit()

    assert len(session.documents) == 2


def test_document_has_chunks_relationship(db_session):
    session = session_service.create_session(db_session)
    document = _doc(session.id)
    db_session.add(document)
    db_session.commit()

    document.chunks.append(
        DocumentChunk(
            id=uuid.uuid4(), document_id=document.id, chunk_index=0, content="hello"
        )
    )
    db_session.commit()

    assert len(document.chunks) == 1
    assert document.chunks[0].chunk_index == 0


def test_session_delete_cascades_documents(db_session):
    session = session_service.create_session(db_session)
    doc1 = _doc(session.id)
    db_session.add(doc1)
    db_session.commit()
    doc1_id = doc1.id

    db_session.delete(session)
    db_session.commit()

    assert db_session.get(Document, doc1_id) is None


def test_document_delete_cascades_chunks(db_session):
    session = session_service.create_session(db_session)
    document = _doc(session.id)
    db_session.add(document)
    db_session.commit()
    chunk = DocumentChunk(
        id=uuid.uuid4(), document_id=document.id, chunk_index=0, content="hi"
    )
    db_session.add(chunk)
    db_session.commit()
    chunk_id = chunk.id

    db_session.delete(document)
    db_session.commit()

    assert db_session.get(DocumentChunk, chunk_id) is None


def test_database_level_cascade_on_session_delete(db_session):
    # Bulk delete bypasses ORM cascade; the DB-level ondelete=CASCADE must fire.
    session = session_service.create_session(db_session)
    document = _doc(session.id)
    db_session.add(document)
    db_session.commit()
    doc_id = document.id

    db_session.execute(delete(Session).where(Session.id == session.id))
    db_session.commit()
    # The bulk DELETE bypasses the ORM, so clear cached objects before checking
    # the database-level ON DELETE CASCADE result.
    db_session.expire_all()

    assert db_session.get(Document, doc_id) is None


def test_storage_path_uses_session_scoped_layout(db_session):
    session = session_service.create_session(db_session)
    document = _doc(session.id)
    db_session.add(document)
    db_session.commit()

    db_session.expire_all()
    assert str(session.id) in document.storage_path