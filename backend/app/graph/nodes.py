"""LangGraph node definitions.

Future responsibilities (later phase):
- rewrite_query: improve a weak query before retrieval
- retrieve: fetch evidence for the (possibly rewritten) query
- grade_retrieval: decide whether retrieved evidence is sufficient
- generate: produce a grounded answer from evidence
- critique_answer: judge whether the answer is faithful to the evidence
"""


def rewrite_query(state):
    """TODO (later phase): query-rewrite node. Not implemented in Phase 0."""
    raise NotImplementedError


def retrieve(state):
    """TODO (later phase): retrieval node. Not implemented in Phase 0."""
    raise NotImplementedError


def grade_retrieval(state):
    """TODO (later phase): retrieval-grading node. Not implemented in Phase 0."""
    raise NotImplementedError


def generate(state):
    """TODO (later phase): generation node. Not implemented in Phase 0."""
    raise NotImplementedError


def critique_answer(state):
    """TODO (later phase): answer-criticism node. Not implemented in Phase 0."""
    raise NotImplementedError
