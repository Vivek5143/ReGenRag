"""Prompts for LLM-based evaluation judges.

Each module builds a prompt that asks the LLM to produce a single JSON
object with a ``score`` (0.0–1.0) and a ``reason`` string. The JSON is
extracted from arbitrary prose using the same robust strategy as the
relevance grader (first ``{`` to last ``}`).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Grounding / Faithfulness Judge
# ---------------------------------------------------------------------------

_GROUNDING_SYSTEM = (
    "You are an expert RAG quality judge. Your job is to evaluate whether "
    "a generated answer is faithful to — i.e. fully supported by — the "
    "provided evidence snippets. You do NOT use any outside knowledge. "
    "If a claim in the answer cannot be verified from the evidence, "
    "treat it as unsupported."
)

_GROUNDING_INSTRUCTIONS = (
    "Score the answer on the following scale:\n"
    "- 1.0: Every claim is directly supported by the evidence.\n"
    "- 0.7: Most claims are supported; a few stretch beyond the evidence.\n"
    "- 0.4: Some claims are supported but significant parts are unsupported.\n"
    "- 0.0: The answer is largely hallucinated or not based on the evidence."
)

_GROUNDING_FORMAT = (
    '{"score": 1.0, "reason": "one sentence explaining the score"}'
)


def build_grounding_prompt(
    question: str,
    evidence: str,
    answer: str,
) -> str:
    """Build the grounding/faithfulness evaluation prompt."""

    return (
        f"{_GROUNDING_SYSTEM}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{evidence}\n\n"
        f"Generated answer:\n{answer}\n\n"
        f"{_GROUNDING_INSTRUCTIONS}\n\n"
        f"Respond with ONLY a single JSON object, exactly in this form:\n"
        f"{_GROUNDING_FORMAT}"
    )


# ---------------------------------------------------------------------------
# Answer Correctness Judge
# ---------------------------------------------------------------------------

_CORRECTNESS_SYSTEM = (
    "You are an expert answer-quality judge. Your job is to evaluate whether "
    "a generated answer is semantically correct compared to an expected answer. "
    "You judge semantic equivalence, not exact wording. If the generated answer "
    "conveys the same meaning as the expected answer, score it highly."
)

_CORRECTNESS_INSTRUCTIONS = (
    "Score the answer on the following scale:\n"
    "- 1.0: Fully correct — conveys the same meaning as the expected answer.\n"
    "- 0.7: Mostly correct — key information is right but some details differ.\n"
    "- 0.4: Partially correct — some facts are right, others are wrong.\n"
    "- 0.0: Incorrect — does not match the expected answer in meaning."
)

_CORRECTNESS_FORMAT = (
    '{"score": 1.0, "reason": "one sentence explaining the score"}'
)


def build_correctness_prompt(
    question: str,
    expected_answer: str,
    generated_answer: str,
) -> str:
    """Build the answer-correctness evaluation prompt."""

    return (
        f"{_CORRECTNESS_SYSTEM}\n\n"
        f"Question:\n{question}\n\n"
        f"Expected answer:\n{expected_answer}\n\n"
        f"Generated answer:\n{generated_answer}\n\n"
        f"{_CORRECTNESS_INSTRUCTIONS}\n\n"
        f"Respond with ONLY a single JSON object, exactly in this form:\n"
        f"{_CORRECTNESS_FORMAT}"
    )
