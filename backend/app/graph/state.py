"""Graph state shared across LangGraph nodes.

The eventual self-healing loop threads this state through nodes that:

    rewrite_query -> retrieve -> grade_retrieval -> generate -> critique_answer

and retries (up to max_retries) when retrieval or grounding is poor.

The graph itself is not built yet — this is the state contract only.
"""

from typing import TypedDict


class RagState(TypedDict, total=False):
    """Mutable state carried through the RAG workflow."""

    question: str
    rewritten_query: str
    # Retrieved evidence documents (list of chunk documents with metadata).
    retrieved_documents: list
    # Score from the retrieval grader (or raw retrieval signal).
    retrieval_score: float | None
    answer: str
    # Score from the answer critic (grounded vs. hallucinated).
    faithfulness_score: float | None
    retry_count: int
    max_retries: int
    # Terminal decision: e.g. "answer", "refuse", "give_up".
    final_decision: str | None
