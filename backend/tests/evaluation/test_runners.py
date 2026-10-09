"""Tests for the evaluation runners."""

import pytest
import uuid
from unittest.mock import MagicMock

from app.evaluation.baseline_runner import run_baseline
from app.evaluation.regenrag_runner import run_regenrag
from app.evaluation.model import EvalCase
from app.evaluation.result import EvalResult
from app.retrieval.vector_store import RetrievedChunk


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def sample_case():
    return EvalCase(
        id=str(uuid.uuid4()),
        query="What is the capital of France?",
        expected_answer="Paris",
        relevant_chunk_ids=[uuid.uuid4()],
        metadata={"category": "A"}
    )


def test_run_baseline(mock_db, sample_case, monkeypatch):
    """Test baseline runner returns expected EvalResult."""
    mock_chunks = [
        RetrievedChunk(
            chunk_id=sample_case.relevant_chunk_ids[0],
            document_id=uuid.uuid4(),
            chunk_index=0,
            page_number=1,
            content="Paris is the capital of France.",
            score=0.9
        )
    ]

    # Mock dependencies
    monkeypatch.setattr("app.retrieval.retriever.retrieve", lambda *args, **kwargs: mock_chunks)
    monkeypatch.setattr("app.generation.generator.generate_answer", lambda q, c: "Paris")

    result = run_baseline(mock_db, sample_case)

    assert isinstance(result, EvalResult)
    assert result.case_id == sample_case.id
    assert result.hit_rate == 1.0
    assert result.mrr == 1.0
    assert result.ndcg == 1.0
    assert result.grounding_score is None
    assert result.rewritten_query is None
    assert result.attempts == 1


def test_run_regenrag(mock_db, sample_case, monkeypatch):
    """Test ReGenRAG runner returns expected EvalResult with healing metadata."""
    from app.services.rag_service import RagAnswer, RagSource

    mock_rag_answer = RagAnswer(
        answer="Paris",
        sources=[
            RagSource(
                document_id=uuid.uuid4(),
                chunk_id=sample_case.relevant_chunk_ids[0],
                chunk_index=0,
                page_number=1,
                score=0.9
            )
        ],
        grounding_score=0.95,
        attempts=1,
        healed=False,
        rewritten_query=None,
        healing_steps=[],
        retry_exhausted=False
    )

    # Mock rag_service.answer_question at the module where it's used
    monkeypatch.setattr("app.services.rag_service.answer_question", lambda *args, **kwargs: mock_rag_answer)

    result = run_regenrag(mock_db, sample_case)

    assert isinstance(result, EvalResult)
    assert result.case_id == sample_case.id
    assert result.hit_rate == 1.0
    assert result.grounding_score == 0.95
    assert result.attempts == 1
    assert result.rewritten_query is None
    assert result.healing_steps == []


def test_run_regenrag_with_recovery(mock_db, sample_case, monkeypatch):
    """Test ReGenRAG runner with a recovered case (rewrite)."""
    from app.services.rag_service import RagAnswer, RagSource, HealingStep, HealingAction

    mock_rag_answer = RagAnswer(
        answer="Paris",
        sources=[
            RagSource(
                document_id=uuid.uuid4(),
                chunk_id=sample_case.relevant_chunk_ids[0],
                chunk_index=0,
                page_number=1,
                score=0.9
            )
        ],
        grounding_score=0.8,
        attempts=2,
        healed=True,
        rewritten_query="Capital city of France",
        healing_steps=[
            HealingStep(attempt=1, action=HealingAction.QUERY_REWRITE, details="Initial retrieval poor")
        ],
        retry_exhausted=False
    )

    monkeypatch.setattr("app.services.rag_service.answer_question", lambda *args, **kwargs: mock_rag_answer)

    result = run_regenrag(mock_db, sample_case)

    assert result.rewritten_query == "Capital city of France"
    assert result.attempts == 2
    assert result.healing_steps == mock_rag_answer.healing_steps
