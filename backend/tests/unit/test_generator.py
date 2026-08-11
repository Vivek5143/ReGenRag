"""Tests for the answer generator.

A fake LLM client records the prompt it received and returns a canned answer,
so no real provider is ever called.
"""

import pytest

from app.core.exceptions import LLMError
from app.generation import generator, prompt


class FakeLLM:
    """Records the last prompt and returns a canned response (or raises)."""

    def __init__(self, response="A fine answer.", error=None):
        self.response = response
        self.error = error
        self.last_prompt = None

    def complete(self, prompt, *, max_tokens=512):
        self.last_prompt = prompt
        if self.error:
            raise self.error
        return self.response


def test_context_and_question_passed_to_llm():
    fake = FakeLLM()
    context = "[Source: Page 4]\nThe report covers X."

    answer = generator.generate_answer("What is covered?", context, client=fake)

    assert fake.last_prompt == prompt.build_baseline_prompt("What is covered?", context)
    assert "The report covers X." in fake.last_prompt
    assert "What is covered?" in fake.last_prompt
    assert answer == "A fine answer."


def test_returns_llm_response_unchanged():
    fake = FakeLLM(response="  Grounded answer text.  ")

    answer = generator.generate_answer("q", "ctx", client=fake)

    assert answer == "Grounded answer text."


def test_llm_error_propagates():
    fake = FakeLLM(error=LLMError("provider down"))

    with pytest.raises(LLMError):
        generator.generate_answer("q", "ctx", client=fake)


def test_unexpected_llm_failure_is_wrapped():
    fake = FakeLLM(error=RuntimeError("boom"))

    with pytest.raises(LLMError):
        generator.generate_answer("q", "ctx", client=fake)
