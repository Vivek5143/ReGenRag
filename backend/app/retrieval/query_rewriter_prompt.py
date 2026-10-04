"""LLM prompt for Phase 4 query rewriting.

The model must reformulate a user query to improve retrieval specificity.
It must not answer the user's question.

Output requirements:
- Return ONLY a single JSON object: {"rewritten_query": "..."}
- The rewritten query must be non-empty.
"""

from __future__ import annotations


def build_query_rewrite_prompt(*, original_query: str, context_snippet: str) -> str:
    """Build a query-rewrite prompt.

    The model is given a short snippet of retrieved content to ground the
    rewrite in what was (insufficiently) found. It is not allowed to invent
    facts outside that context.
    """

    system = (
        "You are an expert query rewriter for semantic search / vector retrieval. "
        "Your job is to rewrite the user's query to be more specific and more "
        "likely to retrieve relevant document chunks. "
        "You must NOT answer the question. "
        "You must NOT add new facts or make up information. "
        "You must preserve the user's original intent."
    )

    instructions = (
        "Use the retrieved context snippet to identify missing specificity and "
        "ambiguities. Improve the query by clarifying key entities, topics, "
        "timeframes, and constraints that matter for retrieval. "
        "Return ONLY the requested JSON."
    )

    json_format = '{"rewritten_query": "one rewritten query string"}'

    return (
        f"{system}\n\n"
        f"Original query:\n{original_query}\n\n"
        f"Retrieved context snippet (may be insufficient evidence):\n{context_snippet}\n\n"
        f"{instructions}\n\n"
        f"Return ONLY a single JSON object exactly in this form:\n{json_format}"
    )
