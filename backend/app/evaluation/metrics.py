"""Evaluation metrics.

Future responsibilities:
- retrieval metrics (e.g. hit rate, MRR, nDCG, retrieval score)
- answer metrics (e.g. faithfulness / grounding score)
- storing results under data/evaluation for offline analysis
"""


def compute_retrieval_metrics(*args, **kwargs):
    """TODO (later phase): retrieval-quality metrics. Not implemented in Phase 0."""
    raise NotImplementedError


def compute_answer_metrics(*args, **kwargs):
    """TODO (later phase): answer-grounding metrics. Not implemented in Phase 0."""
    raise NotImplementedError
