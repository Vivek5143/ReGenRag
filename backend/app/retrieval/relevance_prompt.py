"""LLM retrieval-relevance grading prompt.

Instructs the model to act as a retrieval-quality judge: decide whether a
retrieved chunk contains information that helps answer the user's question.
The model never answers the question and must not use outside knowledge. The
prompt asks for exactly one JSON object so the grader can parse a
machine-readable verdict instead of free-form prose.
"""

_SYSTEM_PROMPT = (
    "You are an expert retrieval-quality judge. You evaluate whether a "
    "retrieved document chunk contains information that is relevant for "
    "answering a user's question. You do NOT answer the question yourself. "
    "Judge ONLY the chunk content supplied below; do not use any outside "
    "knowledge."
)

_JSON_FORMAT = (
    '{"relevant": true, "reason": "one short sentence explaining the judgement"}'
)


def build_relevance_prompt(question: str, chunk: str) -> str:
    """Assemble the LLM relevance-grading prompt for ``question`` and ``chunk``.

    ``chunk`` is the plain text of one retrieved chunk. A chunk is relevant
    when it contains information that could directly help answer the question;
    related-but-insufficient and irrelevant content are both not relevant. The
    response must be a single JSON object; anything else is unparseable and the
    grader treats the chunk as not relevant.
    """
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved chunk:\n{chunk}\n\n"
        "Is the information in this chunk relevant to answering the question? "
        "A chunk is relevant only when it contains information that could "
        "directly help answer the question. Related-but-insufficient "
        "information and irrelevant information are both NOT relevant. Merely "
        "sharing keywords or the same general topic does NOT make a chunk "
        "relevant.\n\n"
        f"Respond with ONLY a single JSON object, exactly in this form:\n"
        f"{_JSON_FORMAT}"
    )
