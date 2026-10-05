"""Tests for the Phase 6 RAG service orchestration.

These tests mock retrieval, LLM relevance grading, query rewriting, context
building, answer generation, and grounding evaluation.

Phase 4 step 2/3 responsibilities:
- Similarity grading gates generation.
- When similarity is insufficient, we rewrite once and then re-retrieve once.
- If the second evidence is still insufficient, we return the controlled
  insufficient-evidence response.

Phase 5:
- Grounding evaluation of generated answers.

Phase 6:
- Bounded self-healing retry loop for retrieval and grounding failures.
"""

from __future__ import annotations

import uuid

import pytest

from app.core.config import settings
from app.core.exceptions import NoProcessedDocumentError, SessionNotFoundError
from app.db.enums import DocumentStatus
from app.db.models.document import Document
from app.evaluation import grounding as grounding_module
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


def _grounding_result(score: float = 1.0) -> grounding_module.GroundingResult:
    """Return a fake grounding result with the given score."""
    return grounding_module.GroundingResult(score=score, reason="supported")


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

    # Grounding evaluation: return high score so no healing is triggered.
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
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
    assert result.healed is False
    assert result.attempts == 1
    assert result.healing_steps == []
    assert result.grounding_score == 1.0
    assert result.retry_exhausted is False

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

    # Grounding evaluation: return high score for the second attempt so no healing.
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

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
    assert result.healed is True  # Recovery occurred
    assert result.attempts == 2
    assert len(result.healing_steps) >= 2  # QUERY_REWRITE + RE_RETRIEVE
    assert result.grounding_score == 1.0
    assert result.retry_exhausted is False

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
    """Test C: first insufficient => rewrite => second insufficient => STOP.

    With max_rag_retries=2, we get up to 3 attempts (initial + 2 retries).
    The second retry uses the same rewritten query since rewrite returns the same string.
    """

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

    # LLM relevance grading runs for each retrieval attempt.
    llm_grade_calls = {"count": 0}

    def fake_grade_relevance(question, chunks, **_):
        llm_grade_calls["count"] += 1
        if chunks == initial_chunks:
            return _llm_grading(has_relevant_evidence=True)
        return _llm_grading(has_relevant_evidence=False)

    monkeypatch.setattr(rag_service.llm_grader, "grade_relevance", fake_grade_relevance)

    # Grounding evaluation won't be called because retrieval is insufficient,
    # but we mock it to be safe.
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

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
    assert result.healed is False
    assert result.retry_exhausted is True
    assert len(result.healing_steps) >= 3  # QUERY_REWRITE + RE_RETRIEVE + RETRY_EXHAUSTED (possibly more)

    # With max_rag_retries=2: attempt 0 (initial), attempt 1 (retry 1), attempt 2 (retry 2)
    # That's 3 retrieval calls
    assert retrieve_calls["count"] == 3
    assert retrieve_calls["questions"] == [original_question, "Rewritten query", "Rewritten query"]

    # Generator not called.
    assert llm_grade_calls["count"] == 3

    # Sources correspond to LAST retrieval.
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

    # Grounding evaluation won't be called because retrieval is insufficient,
    # but we mock it to be safe.
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
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
    assert result.healed is False
    assert result.retry_exhausted is True
    assert len(result.healing_steps) >= 1  # QUERY_REWRITE + RETRY_EXHAUSTED

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

    # Grounding evaluation won't be called because retrieval is insufficient,
    # but we mock it to be safe.
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
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
    assert result.healed is False
    assert result.retry_exhausted is True

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

    # Grounding evaluation won't be called because retrieval is NO_RESULTS,
    # but we mock it to be safe.
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
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
    assert result.healed is False
    assert result.retry_exhausted is True

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

    # Grounding evaluation won't be called because retrieval is NO_RESULTS,
    # but we mock it to be safe.
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
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
    assert result.healed is False
    assert result.retry_exhausted is True

    # Exactly one retrieval; the second (rewrite-driven) retrieval never fires.
    assert retrieve_calls["count"] == 1
    assert retrieve_calls["questions"] == [original_question]

    # Rewriting is skipped on the NO_RESULTS path.
    assert rewrite_calls == []

    # Generation is gated off.
    assert generate_calls == []


# =============================================================================
# Phase 6 Self-Healing Tests
# =============================================================================


def test_first_attempt_succeeds_no_healing(db_session, monkeypatch):
    """Test 1: First attempt succeeds - retrieve -> grade -> generate -> grounding PASS."""
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

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=True),
    )

    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    monkeypatch.setattr(
        rag_service.query_rewriter,
        "rewrite_query",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("query rewriter should not be called")
        ),
    )

    def fake_build_context(chunks):
        return "CONTEXT_FROM_FIRST"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    def fake_generate_answer(question, context):
        return "Generated answer."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == "Generated answer."
    assert result.healed is False
    assert result.attempts == 1
    assert result.healing_steps == []
    assert result.grounding_score == 1.0
    assert result.retry_exhausted is False
    assert retrieve_calls["count"] == 1


def test_retrieval_failure_then_successful_recovery(db_session, monkeypatch):
    """Test 2: Retrieval failure -> successful recovery on retry."""
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

    llm_grade_calls = {"count": 0}

    def fake_grade_relevance(question, chunks, **_):
        llm_grade_calls["count"] += 1
        if chunks == initial_chunks:
            return _llm_grading(has_relevant_evidence=False)
        return _llm_grading(has_relevant_evidence=True)

    monkeypatch.setattr(rag_service.llm_grader, "grade_relevance", fake_grade_relevance)

    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    def fake_rewrite(*args, **kwargs):
        return "Rewritten query"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    def fake_build_context(chunks):
        return "CONTEXT_FROM_SECOND"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    def fake_generate_answer(question, context):
        return "Final answer from second retrieval."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == "Final answer from second retrieval."
    assert result.healed is True
    assert result.attempts == 2
    assert result.grounding_score == 1.0
    assert result.retry_exhausted is False
    assert len(result.healing_steps) >= 2
    # Verify healing action types
    actions = [step.action for step in result.healing_steps]
    assert rag_service.HealingAction.QUERY_REWRITE in actions
    assert rag_service.HealingAction.RE_RETRIEVE in actions


def test_retrieval_failure_then_retry_exhaustion(db_session, monkeypatch):
    """Test 3: Retrieval failure -> retry exhaustion (all attempts fail)."""
    session = _with_processed_document(db_session)
    original_question = "What is this?"

    initial_chunks = [
        _fake_chunk(1, "initial", 0, score=0.3),
    ]
    second_chunks = [
        _fake_chunk(3, "second", 0, score=0.3),
    ]
    third_chunks = [
        _fake_chunk(5, "third", 0, score=0.3),
    ]

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        assert session_id == session.id
        if question == original_question:
            return initial_chunks
        if question == "Rewritten query 1":
            return second_chunks
        if question == "Rewritten query 2":
            return third_chunks
        raise AssertionError(f"unexpected retrieval question: {question!r}")

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    def fake_grade_relevance(question, chunks, **_):
        return _llm_grading(has_relevant_evidence=False)

    monkeypatch.setattr(rag_service.llm_grader, "grade_relevance", fake_grade_relevance)

    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    rewrite_count = {"count": 0}

    def fake_rewrite(*args, **kwargs):
        rewrite_count["count"] += 1
        return f"Rewritten query {rewrite_count['count']}"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    monkeypatch.setattr(
        rag_service.generator,
        "generate_answer",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("generator.generate_answer should not be called")
        ),
    )

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.healed is False
    assert result.retry_exhausted is True
    # With max_retries=2, we should have at most 3 attempts (initial + 2 retries)
    assert result.attempts <= 3
    assert retrieve_calls["count"] <= 3
    # Should have RETRY_EXHAUSTED action
    actions = [step.action for step in result.healing_steps]
    assert rag_service.HealingAction.RETRY_EXHAUSTED in actions


def test_grounding_failure_then_successful_recovery(db_session, monkeypatch):
    """Test 4: Grounding failure -> successful recovery on retry."""
    session = _with_processed_document(db_session)
    original_question = "What is the answer?"

    initial_chunks = [
        _fake_chunk(1, "relevant evidence", 0, score=0.9),
    ]

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        assert session_id == session.id
        return initial_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=True),
    )

    # First grounding fails, second passes
    grounding_calls = {"count": 0}

    def fake_grade_grounding(question, evidence, answer, **_):
        grounding_calls["count"] += 1
        if grounding_calls["count"] == 1:
            return _grounding_result(0.3)  # Below threshold
        return _grounding_result(1.0)  # Above threshold

    monkeypatch.setattr(rag_service.grounding, "grade_grounding", fake_grade_grounding)

    def fake_rewrite(*args, **kwargs):
        return "Rewritten query for grounding"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    def fake_build_context(chunks):
        return "CONTEXT"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    generate_calls = {"count": 0}

    def fake_generate_answer(question, context):
        generate_calls["count"] += 1
        return f"Answer attempt {generate_calls['count']}"

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.healed is True
    assert result.attempts == 2
    assert result.grounding_score == 1.0
    assert result.retry_exhausted is False
    # Should have ANSWER_REGENERATE and CONTEXT_IMPROVEMENT actions
    actions = [step.action for step in result.healing_steps]
    assert rag_service.HealingAction.ANSWER_REGENERATE in actions
    assert rag_service.HealingAction.CONTEXT_IMPROVEMENT in actions
    assert rag_service.HealingAction.RE_RETRIEVE in actions


def test_grounding_failure_then_retry_exhaustion(db_session, monkeypatch):
    """Test 5: Repeated grounding failure terminates at retry limit."""
    session = _with_processed_document(db_session)
    original_question = "What is the answer?"

    initial_chunks = [
        _fake_chunk(1, "relevant evidence", 0, score=0.9),
    ]

    retrieve_calls = {"count": 0, "questions": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["questions"].append(question)
        assert session_id == session.id
        return initial_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=True),
    )

    # All grounding attempts fail
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(0.3),
    )

    def fake_rewrite(*args, **kwargs):
        return "Rewritten query"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    def fake_build_context(chunks):
        return "CONTEXT"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    def fake_generate_answer(question, context):
        return "Ungrounded answer"

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.healed is False
    assert result.retry_exhausted is True
    assert result.grounding_score == 0.3
    # With max_retries=2, we should have at most 3 attempts
    assert result.attempts <= 3
    assert retrieve_calls["count"] <= 3
    actions = [step.action for step in result.healing_steps]
    assert rag_service.HealingAction.RETRY_EXHAUSTED in actions


def test_retry_limit_enforcement(db_session, monkeypatch):
    """Test 6: Mock components so every attempt fails, assert attempts <= MAX_RETRIES + 1."""
    session = _with_processed_document(db_session)
    original_question = "What is this?"

    # Always return insufficient chunks
    def fake_retrieve(db, session_id, question, top_k=None):
        return [_fake_chunk(1, "weak evidence", 0, score=0.3)]

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=False),
    )

    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    def fake_rewrite(*args, **kwargs):
        return "Rewritten query"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    monkeypatch.setattr(
        rag_service.generator,
        "generate_answer",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("generator.generate_answer should not be called")
        ),
    )

    result = rag_service.answer_question(db_session, session.id, original_question)

    # With max_rag_retries=2, max attempts = 3 (initial + 2 retries)
    max_attempts = rag_service.settings.max_rag_retries + 1 if hasattr(rag_service, 'settings') else 3
    # Access settings from the module
    from app.core.config import settings as app_settings
    max_attempts = app_settings.max_rag_retries + 1
    assert result.attempts <= max_attempts
    assert result.retry_exhausted is True


def test_no_unnecessary_healing(db_session, monkeypatch):
    """Test 7: When retrieval and grounding are successful, no unnecessary calls."""
    session = _with_processed_document(db_session)
    original_question = "What is this?"

    initial_chunks = [
        _fake_chunk(4, "alpha", 0, score=0.9),
        _fake_chunk(7, "beta", 1, score=0.8),
    ]

    retrieve_calls = {"count": 0}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        return initial_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=True),
    )

    grounding_calls = {"count": 0}

    def fake_grade_grounding(question, evidence, answer, **_):
        grounding_calls["count"] += 1
        return _grounding_result(1.0)

    monkeypatch.setattr(rag_service.grounding, "grade_grounding", fake_grade_grounding)

    # These should NOT be called
    rewriter_called = {"called": False}

    def fake_rewrite(*args, **kwargs):
        rewriter_called["called"] = True
        return "Should not be called"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    def fake_build_context(chunks):
        return "CONTEXT"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    def fake_generate_answer(question, context):
        return "Generated answer."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.healed is False
    assert result.attempts == 1
    assert not rewriter_called["called"]
    assert retrieve_calls["count"] == 1
    assert grounding_calls["count"] == 1
    assert result.healing_steps == []


def test_healing_action_classification(db_session, monkeypatch):
    """Test 8: Different failure categories result in expected healing actions."""
    # Test LOW_SIMILARITY -> QUERY_REWRITE
    session = _with_processed_document(db_session)
    original_question = "What is this?"

    initial_chunks = [
        _fake_chunk(1, "initial", 0, score=0.3),
    ]
    second_chunks = [
        _fake_chunk(2, "second", 0, score=0.9),
    ]

    retrieve_calls = {"count": 0, "chunks": []}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        retrieve_calls["chunks"].append(question)
        assert session_id == session.id
        if retrieve_calls["count"] == 1:
            return initial_chunks
        return second_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=True),
    )

    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    def fake_rewrite(*args, **kwargs):
        return "Rewritten query"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    def fake_build_context(chunks):
        return "CONTEXT"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    def fake_generate_answer(question, context):
        return "Generated answer."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    # LOW_SIMILARITY should trigger QUERY_REWRITE
    actions = [step.action for step in result.healing_steps]
    assert rag_service.HealingAction.QUERY_REWRITE in actions
    assert rag_service.HealingAction.RE_RETRIEVE in actions
    assert result.healed is True


# =============================================================================
# Phase 6 Retrieval Gate Regression Tests
# These tests verify the Phase 4 retrieval gate design:
# - Similarity grading is the ONLY retrieval sufficiency gate
# - LLM relevance grading is informational only (does not gate generation)
# - No fallback bypasses the retrieval gate
# =============================================================================


def test_retrieval_gate_similarity_only(db_session, monkeypatch):
    """Case 4: All required retrieval checks PASS → Retrieval PASS → Generation allowed.

    In Phase 4 design, the ONLY retrieval gate is similarity_grader.sufficient.
    LLM relevance is computed but does NOT affect sufficiency.
    """
    session = _with_processed_document(db_session)
    original_question = "What is this?"

    # High similarity chunks = similarity PASS
    initial_chunks = [
        _fake_chunk(1, "highly relevant content", 0, score=0.9),
    ]

    retrieve_calls = {"count": 0}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        return initial_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    # LLM relevance FAIL (returns False) - but this should NOT block generation
    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=False),
    )

    # Grounding passes
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    def fake_build_context(chunks):
        return "CONTEXT"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    def fake_generate_answer(question, context):
        return "Generated answer."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    # Generation should succeed because similarity gate PASSED
    # LLM relevance FAIL is informational only
    assert result.answer == "Generated answer."
    assert result.healed is False
    assert result.attempts == 1
    assert result.retrieval_grading.sufficient is True
    assert result.llm_relevance.has_relevant_evidence is False


def test_retrieval_gate_similarity_fails_llm_relevance_passes(db_session, monkeypatch):
    """Case 3: Similarity FAIL + LLM relevance PASS → Retrieval FAIL → Healing triggered.

    Even if LLM says chunks are relevant, if similarity gate fails, healing occurs.
    """
    session = _with_processed_document(db_session)
    original_question = "What is this?"

    # Low similarity chunks = similarity FAIL
    initial_chunks = [
        _fake_chunk(1, "low similarity content", 0, score=0.3),
    ]
    # Second retrieval returns high similarity
    second_chunks = [
        _fake_chunk(2, "high similarity content", 0, score=0.9),
    ]

    retrieve_calls = {"count": 0}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        if retrieve_calls["count"] == 1:
            return initial_chunks
        return second_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    # LLM relevance PASS for both attempts
    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=True),
    )

    # Grounding passes
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    def fake_rewrite(*args, **kwargs):
        return "Rewritten query"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    def fake_build_context(chunks):
        return "CONTEXT"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    def fake_generate_answer(question, context):
        return "Generated answer."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    # Healing should trigger because similarity gate FAILED on first attempt
    assert result.healed is True
    assert result.attempts == 2
    actions = [step.action for step in result.healing_steps]
    assert rag_service.HealingAction.QUERY_REWRITE in actions
    assert rag_service.HealingAction.RE_RETRIEVE in actions


def test_retrieval_gate_no_results_immediate_failure(db_session, monkeypatch):
    """Case: NO_RESULTS (zero chunks) → Immediate controlled failure, no rewrite.

    Zero chunks retrieved is NOT confused with low-quality chunks.
    No query rewrite, no second retrieval.
    """
    session = _with_processed_document(db_session)
    original_question = "Anything?"

    retrieve_calls = {"count": 0}

    def fake_retrieve(db, session_id, question, top_k=None):
        retrieve_calls["count"] += 1
        return []

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=False),
    )

    # This should NOT be called
    monkeypatch.setattr(
        rag_service.query_rewriter,
        "rewrite_query",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("rewrite_query should not be called for NO_RESULTS")
        ),
    )

    monkeypatch.setattr(
        rag_service.generator,
        "generate_answer",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("generate_answer should not be called for NO_RESULTS")
        ),
    )

    result = rag_service.answer_question(db_session, session.id, original_question)

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.failure_category == RetrievalFailureCategory.NO_RESULTS
    assert result.rewritten_query is None
    assert retrieve_calls["count"] == 1
    assert result.retry_exhausted is True


def test_retrieval_gate_no_fallback_generation(db_session, monkeypatch):
    """Case 5: Retrieval remains insufficient after max retries → No answer generation.

    There is NO fallback path that generates an answer from insufficient evidence.
    """
    session = _with_processed_document(db_session)
    original_question = "What is this?"

    # Always return insufficient chunks (similarity FAIL)
    def fake_retrieve(db, session_id, question, top_k=None):
        return [_fake_chunk(1, "weak evidence", 0, score=0.3)]

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=False),
    )

    def fake_rewrite(*args, **kwargs):
        return "Rewritten query"

    monkeypatch.setattr(rag_service.query_rewriter, "rewrite_query", fake_rewrite)

    # Generator should NEVER be called because retrieval never passes
    monkeypatch.setattr(
        rag_service.generator,
        "generate_answer",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("generate_answer should not be called when retrieval fails")
        ),
    )

    # Grounding won't be called either, but mock to be safe
    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    result = rag_service.answer_question(db_session, session.id, original_question)

    # Must return controlled failure, not a generated answer
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.retry_exhausted is True
    assert result.healed is False


def test_retrieval_gate_llm_relevance_informational_only(db_session, monkeypatch):
    """Verify LLM relevance result is recorded but does not gate retrieval.

    The llm_relevance object is preserved in the response for observability
    but does not change the similarity-based sufficiency decision.
    """
    session = _with_processed_document(db_session)
    original_question = "What is this?"

    initial_chunks = [
        _fake_chunk(1, "content", 0, score=0.9),  # Similarity PASS
    ]

    def fake_retrieve(db, session_id, question, top_k=None):
        return initial_chunks

    monkeypatch.setattr(rag_service.retriever, "retrieve", fake_retrieve)

    # LLM relevance can be anything - it's informational
    monkeypatch.setattr(
        rag_service.llm_grader,
        "grade_relevance",
        lambda q, chunks, **_: _llm_grading(has_relevant_evidence=False),  # FAIL
    )

    monkeypatch.setattr(
        rag_service.grounding,
        "grade_grounding",
        lambda q, evidence, answer, **_: _grounding_result(1.0),
    )

    def fake_build_context(chunks):
        return "CONTEXT"

    monkeypatch.setattr(rag_service.context_builder, "build_context", fake_build_context)

    def fake_generate_answer(question, context):
        return "Generated answer."

    monkeypatch.setattr(rag_service.generator, "generate_answer", fake_generate_answer)

    result = rag_service.answer_question(db_session, session.id, original_question)

    # Generation succeeds despite LLM relevance FAIL
    assert result.answer == "Generated answer."
    assert result.retrieval_grading.sufficient is True
    assert result.llm_relevance.has_relevant_evidence is False
    assert result.llm_relevance.relevant_count == 0
    # The llm_relevance data is preserved for observability
    assert result.llm_relevance is not None
