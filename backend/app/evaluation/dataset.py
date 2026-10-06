"""Dataset loader and validation for Phase 7 evaluation.

Provides helpers to load evaluation cases from JSON files and validate
their structure against the ``EvalCase`` Pydantic model.

The dataset JSON format is a list of case objects with the following fields:

Required fields:
- ``id`` (str): unique case identifier
- ``query`` (str): the question to ask the RAG system
- ``category`` (str): one of ``A``, ``B``, ``C``, ``D``, ``E``, ``F``

Optional fields (populated from ``EvalCase`` model):
- ``expected_answer`` (str, optional): reference answer for correctness checks
- ``relevant_document_ids`` (list[str], optional): oracle document UUIDs
- ``relevant_chunk_ids`` (list[str], optional): oracle chunk UUIDs
- ``difficulty`` (str, optional): ``easy``, ``medium``, ``hard``
- ``metadata`` (dict, optional): arbitrary extra information

Example JSON structure:
.. code-block:: json

    [
      {
        "id": "case-001",
        "query": "What is the capital of France?",
        "expected_answer": "Paris",
        "relevant_chunk_ids": ["e3b0c442-98fc-1c14-9afb-4c8996fb9242"],
        "category": "A",
        "difficulty": "easy",
        "metadata": {"source_doc": "doc-001"}
      }
    ]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import EvalCase


VALID_CATEGORIES = {"A", "B", "C", "D", "E", "F"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def load_cases(path: Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file.

    Args:
        path: Path to the JSON file containing a list of case objects.

    Returns:
        List of validated ``EvalCase`` objects.

    Raises:
        RuntimeError: If the file cannot be read or parsed.
        ValueError: If any case fails validation (invalid category, missing
            required fields, malformed UUIDs, etc.).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset {path}: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError("Dataset root must be a JSON array (list of cases)")

    cases: list[EvalCase] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Case at index {idx} must be an object")
        try:
            case = _validate_and_create_case(item, idx)
            cases.append(case)
        except Exception as exc:
            raise ValueError(f"Case at index {idx} (id={item.get('id', '?')}): {exc}") from exc

    return cases


def _validate_and_create_case(item: dict[str, Any], index: int) -> EvalCase:
    """Validate a single case dict and construct an ``EvalCase``."""
    # Check required fields
    for field in ("id", "query", "category"):
        if field not in item or not item[field]:
            raise ValueError(f"Missing required field: {field}")

    case_id = str(item["id"]).strip()
    query = str(item["query"]).strip()
    category = str(item["category"]).strip().upper()

    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'; must be one of {sorted(VALID_CATEGORIES)}"
        )

    difficulty = None
    if "difficulty" in item and item["difficulty"]:
        diff = str(item["difficulty"]).strip().lower()
        if diff not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{diff}'; must be one of {sorted(VALID_DIFFICULTIES)}"
            )
        difficulty = diff

    # Build the EvalCase - it will validate UUIDs and other constraints
    return EvalCase(
        id=case_id,
        query=query,
        expected_answer=item.get("expected_answer"),
        relevant_document_ids=item.get("relevant_document_ids"),
        relevant_chunk_ids=item.get("relevant_chunk_ids"),
        metadata={
            "category": category,
            "difficulty": difficulty,
            **(item.get("metadata") or {}),
        },
    )


def save_cases(cases: list[EvalCase], path: Path) -> None:
    """Write evaluation cases to a JSON file (for round-trip / generation)."""
    out = []
    for c in cases:
        out.append(
            {
                "id": c.id,
                "query": c.query,
                "expected_answer": c.expected_answer,
                "relevant_document_ids": (
                    [str(d) for d in c.relevant_document_ids] if c.relevant_document_ids else None
                ),
                "relevant_chunk_ids": (
                    [str(c) for c in c.relevant_chunk_ids] if c.relevant_chunk_ids else None
                ),
                "category": c.metadata.get("category") if c.metadata else None,
                "difficulty": c.metadata.get("difficulty") if c.metadata else None,
                "metadata": {k: v for k, v in (c.metadata or {}).items() if k not in {"category", "difficulty"}},
            }
        )
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def create_sample_dataset(path: Path, num_cases: int = 6) -> None:
    """Create a minimal sample dataset for quick testing.

    Generates one case per category (A-F) with dummy UUIDs and queries.
    Useful for smoke-testing the CLI without a real corpus.
    """
    import uuid

    sample = []
    for i, cat in enumerate(sorted(VALID_CATEGORIES), 1):
        case_id = f"case-{i:03d}"
        difficulty = "easy" if i <= 2 else "medium" if i <= 4 else "hard"
        sample.append(
            {
                "id": case_id,
                "query": f"Sample question for category {cat} ({case_id})",
                "expected_answer": f"Expected answer for {case_id}",
                "relevant_chunk_ids": [str(uuid.uuid4()) for _ in range(2)],
                "metadata": {
                    "generated": True,
                    "category": cat,
                    "difficulty": difficulty,
                },
            }
        )
    save_cases([EvalCase(**s) for s in sample], path)


__all__ = [
    "load_cases",
    "save_cases",
    "create_sample_dataset",
    "VALID_CATEGORIES",
    "VALID_DIFFICULTIES",
]