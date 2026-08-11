"""Baseline RAG prompt.

Deliberately simple and self-contained: the model is told to ground its answer
in the supplied context, refuse to invent facts, and say when the information
is not available. No grading, criticism, or self-healing logic here — this is
the baseline that later phases will compare against.
"""

_SYSTEM_PROMPT = (
    "You are a precise, evidence-grounded assistant. Answer the user's question "
    "using ONLY the context provided below. Do not invent facts that are not "
    "supported by the context. If the context does not contain enough "
    "information to answer the question, say the information is not available "
    "in the provided document. When you use information from the context, cite "
    "the source page shown in its [Source: Page N] marker."
)

_EMPTY_CONTEXT = "[no relevant content was retrieved for this question]"


def build_baseline_prompt(question: str, context: str) -> str:
    """Assemble the full baseline RAG prompt for ``question`` and ``context``.

    ``context`` is the output of the context builder (possibly empty). The
    system instructions plus a clearly delimited context and question are
    combined into a single user prompt.
    """
    context_block = context if context else _EMPTY_CONTEXT
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
