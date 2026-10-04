"""Tests for the Phase 4 RAG service orchestration.

These tests mock retrieval, LLM relevance grading, query rewriting, context
building, and answer generation.

Phase 4 step 2/3 responsibilities:
- Similarity grading gates generation.
- When similarity is insufficient, we rewrite once and then re-retrieve once.
- If the second evidence is still insufficient, we return the controlled
  insufficient-evidence response.
"""

from __future__ import annotations

import uuid

import pytest

from app.core.exceptions import NoProcessedDocumentError, SessionNotFoundError
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.retrieval import llm_grader as llm_grader_module
from app.retrieval.failure_classifier import RetrievalFailureCategory
from app.retrieval.similarity_grader import RetrievalGrading
from app.retrieval.vector_store import RetrievedChunk
from app.services import rag_service, session_service


INSUFFICIENT_EVIDENCE_ANSWER = (
    "I couldn't find enough relevant information in the provided documents "
    "to answer this question reliably."
)


def _binding_error_session(db_session):
    """A random id that is guaranteed not to exist."""

    return uuid.uuid4()


def _with_processed_document(db_session):
    session = session_service.create_session(db_session)
    db_session.add(
        Document(
            id=uuid.uuid4(),
            session_id=session.id,
            filename="report.pdf",
            stored_filename="x.pdf",
            content_type="application/pdf",
            file_size=5,
            storage_path=f"{session.id}/report.pdf",
            status=DocumentStatus.PROCESSED,
        )
    )
    db_session.commit()
    return session


def _fake_chunk(page, text, index, *, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=index,
        page_number=page,
        content=text,
        score=score,
    )


def _llm_grading(*, has_relevant_evidence: bool) -> llm_grader_module.LlmRetrievalGrading:
    return llm_grader_module.LlmRetrievalGrading(
        judgements=[],
        relevant_count=1 if has_relevant_evidence else 0,
        failed_count=0,
        has_relevant_evidence=has_relevant_evidence,
        reason="ok" if has_relevant_evidence else "no relevant evidence",
    )


def test_invalid_session_raises_not_found(db_session):
    with pytest.raises(SessionNotFoundError):
        rag_service.answer_question(
            db_session, _binding_error_session(db_session), "hi"
        )


def test_no_processed_document_raises(db_session):
    session = session_service.create_session(db_session)  # no documents at all

    with pytest.raises(NoProcessedDocumentError):
        rag_service.answer_question(db_session, session.id, "hi")


def test_blank_question_rejected(db_session):
    session = session_service.create_session(db_session)

    with pytest.raises(ValueError):
        rag_service.answer_question(db_session, session.id, "   ")


def test_first_retrieval_sufficient_no_rewriter_generator_called_once(
    db_session, monkeypatch
):
    """Test A / F: sufficient first retrieval => no rewrite, no 2nd retrieval."""

    session = _with_processed_document(db_session)
    original_question = "What is this?"

    initial_chunks = [
        _fake_chunk(4, "alpha", 0, score=0.9),
        _fake_chunk(7, "beta", 1, score=0.8),
    ]

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        assert session_id == session.id
        return initial_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    # LLM relevance grading: doesn't affect sufficiency (similarity gate), but
    # it must still run.
    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=True),
    )

    # Ensure query rewriter is not called.
    monkeypatch.setattr(
        rag_service.query_rewriter,
        "rewrite_query",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("query rewriter should not be called")
        ),
    )

    context_calls = {"count": 0, "chunks": None}

    def fake_build_context(chunks):
        context_calls["count"] += 1
        context_calls["chunks"] = chunks
        return "CONTEXT_FROM_FIRST"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    generator_calls = {"count": 0, "question": None, "context": None}

    def fake_generate_answer(question, context):
        generator_calls["count"] += 1
        generator_calls["question"] = question
        generator_calls["context"] = context
        return "Generated answer."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == "Generated answer."
    assert result.rewritten_query is None

    assert retrieve_calls["count"] == 1
    assert retrieve_calls["questions"] == [original_question]

    assert generator_calls["count"] == 1
    assert generator_calls["question"] == original_question
    assert generator_calls["context"] == "CONTEXT_FROM_FIRST"

    assert context_calls["count"] == 1
    assert context_calls["chunks"] == initial_chunks

    assert len(result.sources) == len(initial_chunks)
    assert result.retrieval_grading is not None
    assert result.retrieval_grading.sufficient is True


def test_first_insufficient_rewrite_second_sufficient_generation_uses_second_results(
    db_session, monkeypatch
):
    """Test B: insufficient first => rewrite => second retrieval sufficient."""

    session = _with_processed_document(db_session)
    original_question = "What does the report say?"

    initial_chunks = [
        _fake_chunk(1, "quarterly revenue", 0, score=0.3),
    ]
    second_chunks = [
        _fake_chunk(3, "next fiscal year outlook", 0, score=0.9),
        _fake_chunk(5, "revenue performance summary", 1, score=0.8),
    ]

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        assert session_id == session.id
        if question == original_question:
            return initial_chunks
        if question == "Rewritten query":
            return second_chunks
        raise AssertionError(f"unexpected retrieval question: {question!r}")

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    # LLM relevance grading should run for BOTH retrieval attempts.
    llm_grade_calls = {"count": 0, "chunks_pages": []}

    def fake_grade_relevance(question, chunks, **_):
        llm_grade_calls["count"] += 1
        llm_grade_calls["chunks_pages"].append([c.page_number for c in chunks])

        # Different verdicts per attempt so we know which one is used.
        if chunks == initial_chunks:
            return _llm_grading(has_relevant_evidence=False)
        return _llm_grading(has_relevant_evidence=True)

    monkeypatch.setattr(rag_service.llm_grader, "grade_relevance", fake_grade_relevance)

    # Rewrite query succeeds.
    rewrite_calls = {"count": 0}

    def fake_rewrite(*args, **kwargs):
        rewrite_calls["count"] += 1
        return "Rewritten query"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    context_calls = {"count": 0, "chunks": None}

    def fake_build_context(chunks):
        context_calls["count"] += 1
        context_calls["chunks"] = chunks
        return "CONTEXT_FROM_SECOND"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    generator_calls = {"count": 0, "question": None, "context": None}

    def fake_generate_answer(question, context):
        generator_calls["count"] += 1
        generator_calls["question"] = question
        generator_calls["context"] = context
        return "Final answer from second retrieval."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == "Final answer from second retrieval."
    assert result.rewritten_query == "Rewritten query"

    # Exactly two retrieval calls.
    assert retrieve_calls["count"] == 2
    assert retrieve_calls["questions"] == [original_question, "Rewritten query"]

    # Query rewriter called once.
    assert rewrite_calls["count"] == 1

    # Second retrieval sufficient => generator called once.
    assert generator_calls["count"] == 1
    assert generator_calls["question"] == original_question
    assert generator_calls["context"] == "CONTEXT_FROM_SECOND"

    # Context from SECOND retrieval.
    assert context_calls["count"] == 1
    assert context_calls["chunks"] == second_chunks

    # Sources correspond to SECOND retrieval.
    assert [s.page_number for s in result.sources] == [3, 5]

    # Retrieval + LLM grading used from second attempt.
    assert result.retrieval_grading is not None
    assert result.retrieval_grading.sufficient is True
    assert result.retrieval_grading.best_score == 0.9

    assert result.llm_relevance is not None
    assert result.llm_relevance.has_relevant_evidence is True

    # LLM relevance graded twice (initial + second).
    assert llm_grade_calls["count"] == 2
    assert llm_grade_calls["chunks_pages"][0] == [1]
    assert llm_grade_calls["chunks_pages"][1] == [3, 5]


def test_first_insufficient_rewrite_second_insufficient_returns_controlled_answer(
    db_session, monkeypatch
):
    """Test C: first insufficient => rewrite => second insufficient => STOP."""

    session = _with_processed_document(db_session)
    original_question = "What is this?"

    initial_chunks = [
        _fake_chunk(1, "initial", 0, score=0.3),
    ]
    second_chunks = [
        _fake_chunk(3, "second", 0, score=0.3),
    ]

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        assert session_id == session.id
        if question == original_question:
            return initial_chunks
        if question == "Rewritten query":
            return second_chunks
        raise AssertionError(f"unexpected retrieval question: {question!r}")

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    # LLM relevance grading runs twice.
    llm_grade_calls = {"count": 0}

    def fake_grade_relevance(question, chunks, **_):
        llm_grade_calls["count"] += 1
        if chunks == initial_chunks:
            return _llm_grading(has_relevant_evidence=True)
        return _llm_grading(has_relevant_evidence=False)

    monkeypatch.setattr(rag_service.llm_grader, "grade_relevance", fake_grade_relevance)

    monkeypatch.setattr(
        rag_service.query_rewriter,
        "rewrite_query",
        lambda *a, **k: "Rewritten query",
    )

    monkeypatch.setattr(
        rag_service.generator,
        "generate_answer",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("generator.generate_answer should not be called")
        ),
    )

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.rewritten_query == "Rewritten query"

    assert retrieve_calls["count"] == 2
    assert retrieve_calls["questions"] == [original_question, "Rewritten query"]

    # Generator not called.
    assert llm_grade_calls["count"] == 2

    # Sources correspond to SECOND retrieval.
    assert [s.page_number for s in result.sources] == [3]

    assert result.retrieval_grading is not None
    assert result.retrieval_grading.sufficient is False

    assert result.llm_relevance is not None
    assert result.llm_relevance.has_relevant_evidence is False

    assert result.failure_category == RetrievalFailureCategory.INSUFFICIENT_EVIDENCE


def test_rewriter_returns_none_stops_after_first_retrieval(
    db_session, monkeypatch
):
    """Test D: rewrite returns None => no second retrieval, no generation."""

    session = _with_processed_document(db_session)
    original_question = "What is this?"

    initial_chunks = [
        _fake_chunk(1, "initial", 0, score=0.3),
    ]

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        return initial_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda *a, **k: _llm_grading(has_relevant_evidence=False),
    )

    monkeypatch.setattr(
        rag_service.query_rewriter,
        "rewrite_query",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        rag_service.generator,
        "generate_answer",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("generator.generate_answer should not be called")
        ),
    )

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.rewritten_query is None

    assert retrieve_calls["count"] == 1
    assert retrieve_calls["questions"] == [original_question]

    assert [s.page_number for s in result.sources] == [1]

    assert result.failure_category == RetrievalFailureCategory.LOW_LLM_RELEVANCE


def test_rewriter_returns_empty_string_stops_after_first_retrieval(
    db_session, monkeypatch
):
    """Test E: rewrite returns empty/whitespace => no second retrieval."""

    session = _with_processed_document(db_session)
    original_question = "What is this?"

    initial_chunks = [
        _fake_chunk(1, "initial", 0, score=0.3),
    ]

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        return initial_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda *a, **k: _llm_grading(has_relevant_evidence=False),
    )

    monkeypatch.setattr(
        rag_service.query_rewriter,
        "rewrite_query",
        lambda *a, **k: "  ",
    )

    monkeypatch.setattr(
        rag_service.generator,
        "generate_answer",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("generator.generate_answer should not be called")
        ),
    )

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.rewritten_query is None

    assert retrieve_calls["count"] == 1
    assert retrieve_calls["questions"] == [original_question]

    assert [s.page_number for s in result.sources] == [1]


def test_no_results_classified_and_stops_after_first_retrieval(
    db_session, monkeypatch
):
    """Test F: retriever returns [] => NO_RESULTS, no third retrieval, no generation."""

    session = _with_processed_document(db_session)
    original_question = "Anything?"

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        return []

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda *a, **k: _llm_grading(has_relevant_evidence=False),
    )

    monkeypatch.setattr(
        rag_service.query_rewriter,
        "rewrite_query",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("query rewriter should not be called for NO_RESULTS")
        ),
    )

    monkeypatch.setattr(
        rag_service.generator,
        "generate_answer",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("generator.generate_answer should not be called")
        ),
    )

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.sources == []
    assert result.failure_category == RetrievalFailureCategory.NO_RESULTS
    assert result.rewritten_query is None

    # Exactly one retrieval; no third retrieval.
    assert retrieve_calls["count"] == 1
    assert retrieve_calls["questions"] == [original_question]

    # Grading of an empty retrieval is safe: insufficient, no usable scores.
    assert result.retrieval_grading is not None
    assert result.retrieval_grading.sufficient is False
    assert result.retrieval_grading.best_score is None
    assert result.retrieval_grading.average_score is None


def test_no_results_skips_rewrite_and_second_retrieval(db_session, monkeypatch):
    """NO_RESULTS path: no rewriting, no second retrieval, no generation.

    Distinct from the preceding test: here the rewriter is a recording spy
    (not a throwing assertion), so we can positively confirm it was never
    called and that exactly one retrieval occurred.
    """

    session = _with_processed_document(db_session)
    original_question = "Anything?"

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        return []

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda *a, **k: _llm_grading(has_relevant_evidence=False),
    )

    rewrite_calls = []

    def fake_rewrite(*args, **kwargs):
        rewrite_calls.append((args, kwargs))
        return "should not matter"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    generate_calls = []

    def fake_generate(question, context):
        generate_calls.append((question, context))
        return "should not be generated"

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate)

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.sources == []
    assert result.failure_category == RetrievalFailureCategory.NO_RESULTS
    assert result.rewritten_query is None

    # Exactly one retrieval; the second (rewrite-driven) retrieval never fires.
    assert retrieve_calls["count"] == 1
    assert retrieve_calls["questions"] == [original_question]

    # Rewriting is skipped on the NO_RESULTS path.
    assert rewrite_calls == []

    # Generation is gated off.
    assert generate_calls == []
