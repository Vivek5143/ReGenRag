"""Tests for the LLM-enabled answer evaluator integration (Phase 5).

Verifies that evaluate_case() correctly wires together retrieval metrics,
grounding evaluation, and correctness evaluation — both with and without an
AnswerEvaluator. All tests use a mocked/fake LLM client. No network, no API
keys, no Gemini/Ollama calls.
"""

from __future__ import annotations

import uuid

import pytest

from app.evaluation import AnswerEvaluator, evaluate_case, evaluate_run
from app.evaluation.model import EvalCase
from app.evaluation.result import EvalRunResult
from app.retrieval.vector_store import RetrievedChunk


def _chunk(chunk_id: uuid.UUID, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content="evidence text",
        score=score,
    )


class FakeLLMClient:
    """Records calls and returns canned responses or raises."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses) if responses else []
        self.error = error
        self.calls = []

    def complete(self, prompt, *, max_tokens=512):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return ""


# ---------------------------------------------------------------------------
# evaluate_case WITHOUT LLM evaluator — retrieval only
# ---------------------------------------------------------------------------


def test_evaluate_case_without_evaluator_still_works():
    """Retrieval-only evaluation continues to work when no evaluator is given."""
    chunk_id = uuid.uuid4()
    case = EvalCase(id="q1", query="What is X?", relevant_chunk_ids=[chunk_id])
    retrieved = [_chunk(chunk_id)]

    result = evaluate_case(case, retrieved)

    assert result.hit_rate == 1.0
    assert result.mrr == pytest.approx(1.0)
    assert result.ndcg == pytest.approx(1.0)
    assert result.grounding_score is None
    assert result.answer_quality is None


def test_evaluate_case_without_evaluator_skips_llm_calls():
    """No LLM calls are made when answer_evaluator is None."""
    chunk_id = uuid.uuid4()
    fake = FakeLLMClient(responses=['{"score": 1.0}'])
    case = EvalCase(
        id="q1",
        query="Q",
        relevant_chunk_ids=[chunk_id],
        expected_answer="A",
    )
    # Pass evaluator but no answer/evidence — should not call LLM
    result = evaluate_case(
        case,
        [_chunk(chunk_id)],
        answer_evaluator=AnswerEvaluator(client=fake),
    )
    assert len(fake.calls) == 0
    assert result.grounding_score is None
    assert result.answer_quality is None


# ---------------------------------------------------------------------------
# evaluate_case WITH LLM evaluator
# ---------------------------------------------------------------------------


def test_evaluate_case_with_evaluator_runs_grounding():
    """When answer and evidence are provided, grounding is evaluated."""
    chunk_id = uuid.uuid4()
    case = EvalCase(
        id="q1",
        query="What is X?",
        relevant_chunk_ids=[chunk_id],
        expected_answer="A",
    )
    fake = FakeLLMClient(
        responses=[
            '{"score": 1.0, "reason": "supported"}',  # grounding
            '{"score": 0.8, "reason": "mostly correct"}',  # correctness
        ]
    )
    result = evaluate_case(
        case,
        [_chunk(chunk_id)],
        answer="Yes, it is X.",
        evidence="X is the answer.",
        answer_evaluator=AnswerEvaluator(client=fake),
    )
    assert result.grounding_score == 1.0
    assert result.answer_quality == 0.8
    assert len(fake.calls) == 2


def test_evaluate_case_with_evaluator_missing_expected_answer():
    """Without expected_answer, correctness is skipped but grounding still runs."""
    chunk_id = uuid.uuid4()
    fake = FakeLLMClient(responses=['{"score": 1.0, "reason": "supported"}'])
    case = EvalCase(
        id="q1",
        query="Q",
        relevant_chunk_ids=[chunk_id],
    )
    result = evaluate_case(
        case,
        [_chunk(chunk_id)],
        answer="A",
        evidence="E",
        answer_evaluator=AnswerEvaluator(client=fake),
    )
    assert result.grounding_score == 1.0
    assert result.answer_quality is None  # no expected_answer
    assert len(fake.calls) == 1  # only grounding, not correctness


def test_evaluate_case_with_evaluator_missing_evidence():
    """Without evidence, grounding returns 0.0 and correctness still runs."""
    chunk_id = uuid.uuid4()
    fake = FakeLLMClient(
        responses=['{"score": 0.9, "reason": "correct"}']
    )
    case = EvalCase(
        id="q1",
        query="Q",
        relevant_chunk_ids=[chunk_id],
        expected_answer="A",
    )
    result = evaluate_case(
        case,
        [_chunk(chunk_id)],
        answer="A",
        evidence=None,
        answer_evaluator=AnswerEvaluator(client=fake),
    )
    assert result.grounding_score == 0.0  # no evidence → score 0
    assert result.answer_quality == 0.9
    assert len(fake.calls) == 1  # only correctness


def test_evaluate_case_with_evaluator_no_answer():
    """When answer is None, both grounding and correctness are skipped."""
    chunk_id = uuid.uuid4()
    fake = FakeLLMClient()
    case = EvalCase(
        id="q1",
        query="Q",
        relevant_chunk_ids=[chunk_id],
        expected_answer="A",
    )
    result = evaluate_case(
        case,
        [_chunk(chunk_id)],
        answer=None,
        evidence="E",
        answer_evaluator=AnswerEvaluator(client=fake),
    )
    assert result.grounding_score is None
    assert result.answer_quality is None
    assert len(fake.calls) == 0


def test_evaluate_case_with_evaluator_malformed_grounding_output():
    """Malformed grounding output should not crash the evaluator."""
    chunk_id = uuid.uuid4()
    fake = FakeLLMClient(
        responses=[
            "not valid json",  # grounding fails
            '{"score": 0.7, "reason": "ok"}',  # correctness succeeds
        ]
    )
    case = EvalCase(
        id="q1",
        query="Q",
        relevant_chunk_ids=[chunk_id],
        expected_answer="A",
    )
    result = evaluate_case(
        case,
        [_chunk(chunk_id)],
        answer="A",
        evidence="E",
        answer_evaluator=AnswerEvaluator(client=fake),
    )
    assert result.grounding_score == 0.0  # parse failure → 0.0
    assert result.answer_quality == 0.7
    assert len(fake.calls) == 2


def test_evaluate_case_with_evaluator_malformed_correctness_output():
    """Malformed correctness output should not crash the evaluator."""
    chunk_id = uuid.uuid4()
    fake = FakeLLMClient(
        responses=[
            '{"score": 1.0, "reason": "ok"}',  # grounding succeeds
            "broken json",  # correctness fails
        ]
    )
    case = EvalCase(
        id="q1",
        query="Q",
        relevant_chunk_ids=[chunk_id],
        expected_answer="A",
    )
    result = evaluate_case(
        case,
        [_chunk(chunk_id)],
        answer="A",
        evidence="E",
        answer_evaluator=AnswerEvaluator(client=fake),
    )
    assert result.grounding_score == 1.0
    assert result.answer_quality == 0.0  # parse failure → 0.0
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# evaluate_run aggregation with mixed results
# ---------------------------------------------------------------------------


def test_evaluate_run_with_mixed_grounding_values():
    """avg_grounding averages only over cases that have a score (not None)."""
    chunk_a = uuid.UUID("11111111-1111-1111-1111-111111111111")
    chunk_b = uuid.UUID("22222222-2222-2222-2222-222222222222")
    case_a = EvalCase(
        id="a", query="A", relevant_chunk_ids=[chunk_a],
        expected_answer="A",
    )
    case_b = EvalCase(
        id="b", query="B", relevant_chunk_ids=[chunk_b],
        expected_answer="A",
    )
    result_a = evaluate_case(
        case_a, [_chunk(chunk_a)], answer="A", evidence="E",
        answer_evaluator=AnswerEvaluator(
            client=FakeLLMClient(
                responses=[
                    '{"score": 1.0, "reason": "ok"}',
                    '{"score": 0.8, "reason": "ok"}',
                ]
            )
        ),
    )
    result_b = evaluate_case(
        case_b, [_chunk(chunk_b)], answer="A", evidence="E",
        answer_evaluator=AnswerEvaluator(
            client=FakeLLMClient(
                responses=[
                    '{"score": 0.5, "reason": "ok"}',
                    '{"score": 0.5, "reason": "ok"}',
                ]
            )
        ),
    )

    run = evaluate_run([case_a, case_b], {"a": result_a, "b": result_b})
    assert isinstance(run, EvalRunResult)
    assert run.count == 2
    # Both have grounding scores
    assert run.avg_grounding == pytest.approx(0.75)
    assert run.avg_answer_quality == pytest.approx(0.65)


def test_evaluate_run_with_missing_grounding_values():
    """avg_grounding ignores None values, does not count them as zero."""
    chunk_a = uuid.UUID("11111111-1111-1111-1111-111111111111")
    chunk_b = uuid.UUID("22222222-2222-2222-2222-222222222222")
    case_a = EvalCase(
        id="a", query="A", relevant_chunk_ids=[chunk_a],
        expected_answer="A",
    )
    case_b = EvalCase(
        id="b", query="B", relevant_chunk_ids=[chunk_b],
        expected_answer="A",
    )
    result_a = evaluate_case(
        case_a, [_chunk(chunk_a)], answer="A", evidence="E",
        answer_evaluator=AnswerEvaluator(
            client=FakeLLMClient(
                responses=[
                    '{"score": 1.0, "reason": "ok"}',
                    '{"score": 0.9, "reason": "ok"}',
                ]
            )
        ),
    )
    # case_b has no answer evaluator → no grounding
    result_b = evaluate_case(case_b, [_chunk(chunk_b)])

    run = evaluate_run([case_a, case_b], {"a": result_a, "b": result_b})
    assert run.count == 2
    # Only case_a has grounding_score, so average is 1.0
    assert run.avg_grounding == pytest.approx(1.0)
    # Only case_a has answer_quality, so average is 0.9
    assert run.avg_answer_quality == pytest.approx(0.9)


def test_evaluate_run_all_missing_grounding_returns_zero():
    """When no case has grounding, avg_grounding returns 0.0 (not NaN)."""
    chunk_a = uuid.UUID("11111111-1111-1111-1111-111111111111")
    case_a = EvalCase(id="a", query="A", relevant_chunk_ids=[chunk_a])
    result_a = evaluate_case(case_a, [_chunk(chunk_a)])

    run = evaluate_run([case_a], {"a": result_a})
    assert run.count == 1
    assert run.avg_grounding == 0.0
    assert run.avg_answer_quality == 0.0
