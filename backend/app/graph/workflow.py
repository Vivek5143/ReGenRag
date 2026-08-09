"""LangGraph workflow assembly.

Future responsibilities (later phase):
- wiring the self-healing loop: rewrite -> retrieve -> grade -> generate ->
  critique, with retry edges while retry_count < max_retries
- deciding when to refuse to answer
- exposing a trace of the loop for the UI's Self-Healing Trace view
"""


def build_workflow():
    """TODO (later phase): assemble the LangGraph StateGraph. Not implemented
    in Phase 0.
    """
    raise NotImplementedError
