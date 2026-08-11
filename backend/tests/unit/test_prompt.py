"""Tests for the baseline RAG prompt template."""

from app.generation import prompt


def test_prompt_contains_question_and_context():
    text = prompt.build_baseline_prompt("What is this?", "[Source: Page 1]\nHello")

    assert "What is this?" in text
    assert "[Source: Page 1]" in text
    assert "Hello" in text


def test_prompt_grounds_answer_in_context_only():
    text = prompt.build_baseline_prompt("q", "ctx")

    assert "ONLY" in text
    assert "not available" in text.lower()
    assert "Do not invent" in text


def test_prompt_handles_empty_context():
    text = prompt.build_baseline_prompt("What is this?", "")

    assert "[no relevant content was retrieved for this question]" in text
    assert "What is this?" in text
