"""Answer generation.

Future responsibilities:
- grounding answers strictly in retrieved evidence
- calling the configured LLM provider (LLM_PROVIDER / LLM_MODEL)
- refusing to answer when sufficient evidence cannot be found
- surfacing citations / source references
"""


def generate_answer(*args, **kwargs):
    """Generate a grounded answer from a question and retrieved evidence.

    TODO (later phase): LLM call with evidence context and an explicit refusal
    path when evidence is insufficient. Not implemented in Phase 0.
    """
    raise NotImplementedError
