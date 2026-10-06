import pytest
from pathlib import Path
import json
import uuid

from app.evaluation.dataset import load_cases, VALID_CATEGORIES, VALID_DIFFICULTIES, create_sample_dataset
from app.evaluation.model import EvalCase


def test_valid_dataset(tmp_path: Path):
    """Test loading a valid dataset with all required fields."""
    data = [
        {
            "id": "case-001",
            "query": "Test query",
            "category": "A",
            "expected_answer": "Expected answer",
            "relevant_chunk_ids": [str(uuid.uuid4())],
            "difficulty": "easy",
            "metadata": {"extra": "info"},
        }
    ]
    path = tmp_path / "valid.json"
    path.write_text(json.dumps(data))

    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].id == "case-001"
    assert cases[0].query == "Test query"
    assert cases[0].metadata["category"] == "A"
    assert cases[0].metadata["difficulty"] == "easy"
    assert cases[0].metadata["extra"] == "info"


def test_missing_required_field_id(tmp_path: Path):
    """Missing 'id' field should raise ValueError."""
    data = [{"query": "test", "category": "A"}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Missing required field: id"):
        load_cases(path)


def test_missing_required_field_query(tmp_path: Path):
    """Missing 'query' field should raise ValueError."""
    data = [{"id": "c1", "category": "A"}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Missing required field: query"):
        load_cases(path)


def test_missing_required_field_category(tmp_path: Path):
    """Missing 'category' field should raise ValueError."""
    data = [{"id": "c1", "query": "test"}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Missing required field: category"):
        load_cases(path)


def test_invalid_category(tmp_path: Path):
    """Invalid category should raise ValueError."""
    data = [{"id": "c1", "query": "test", "category": "Z"}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Invalid category"):
        load_cases(path)


def test_invalid_difficulty(tmp_path: Path):
    """Invalid difficulty should raise ValueError."""
    data = [{"id": "c1", "query": "test", "category": "A", "difficulty": "impossible"}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Invalid difficulty"):
        load_cases(path)


def test_malformed_json(tmp_path: Path):
    """Malformed JSON should raise RuntimeError."""
    path = tmp_path / "bad.json"
    path.write_text("{ not valid json }")

    with pytest.raises(RuntimeError, match="Failed to read dataset"):
        load_cases(path)


def test_root_not_array(tmp_path: Path):
    """Root must be a JSON array."""
    path = tmp_path / "bad.json"
    path.write_text('{"id": "c1"}')

    with pytest.raises(ValueError, match="Dataset root must be a JSON array"):
        load_cases(path)


def test_case_not_object(tmp_path: Path):
    """Each case must be an object."""
    data = ["not an object"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="must be an object"):
        load_cases(path)


def test_invalid_uuid_in_chunk_ids(tmp_path: Path):
    """Invalid UUID in relevant_chunk_ids should raise ValueError."""
    data = [{"id": "c1", "query": "test", "category": "A", "relevant_chunk_ids": ["not-a-uuid"]}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError):
        load_cases(path)


def test_multiple_cases(tmp_path: Path):
    """Multiple cases should all be loaded and validated."""
    data = [
        {"id": "c1", "query": "q1", "category": "A"},
        {"id": "c2", "query": "q2", "category": "B", "difficulty": "hard"},
        {"id": "c3", "query": "q3", "category": "C", "expected_answer": "ans"},
    ]
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(data))

    cases = load_cases(path)
    assert len(cases) == 3
    assert cases[0].id == "c1"
    assert cases[1].id == "c2"
    assert cases[2].id == "c3"
    assert cases[1].metadata["difficulty"] == "hard"


def test_category_case_insensitive(tmp_path: Path):
    """Category should be case-insensitive and normalized to uppercase."""
    data = [{"id": "c1", "query": "test", "category": "a"}]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(data))

    cases = load_cases(path)
    assert cases[0].metadata["category"] == "A"


def test_create_sample_dataset(tmp_path: Path):
    """create_sample_dataset should generate valid cases."""
    path = tmp_path / "sample.json"
    create_sample_dataset(path, num_cases=6)

    cases = load_cases(path)
    assert len(cases) == 6
    for c in cases:
        assert c.id.startswith("case-")
        assert c.metadata["category"] in VALID_CATEGORIES
        assert c.metadata["difficulty"] in VALID_DIFFICULTIES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
