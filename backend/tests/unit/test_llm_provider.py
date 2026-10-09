"""Tests for the LLM provider factory and the Gemini client.

All network calls are mocked (httpx.post is stubbed) and provider settings are
monkeypatched, so nothing ever leaves the process.
"""

import httpx
import pytest

from app.core.exceptions import LLMConfigError, LLMError
from app.generation import llm
from app.generation.llm import AnthropicClient, GeminiClient, OpenAICompatibleClient


@pytest.fixture(autouse=True)
def _reset_client_cache(monkeypatch):
    """Isolate the cached client between tests."""
    monkeypatch.setattr(llm, "_client", None)


@pytest.fixture()
def gemini_settings(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "llm_model", "gemini-2.5-flash")
    monkeypatch.setattr(llm.settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(llm.settings, "llm_base_url", "")
    return llm.settings


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RaisingResponse:
    """A fake response whose ``raise_for_status`` raises like httpx does.

    It carries a real ``httpx.Response`` (with Google's error JSON) inside the
    ``HTTPStatusError`` so the client's error-body handling has something to
    read, exactly as in production.
    """

    def __init__(self, status_code, payload):
        self._status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        request = httpx.Request("POST", "https://example.com/v1beta/models/m:generateContent")
        response = httpx.Response(
            self._status_code, json=self._payload, request=request
        )
        raise httpx.HTTPStatusError(
            f"Client error '{self._status_code}' for url '{request.url}'",
            request=request,
            response=response,
        )

    def json(self):
        return self._payload


# --- Provider selection -----------------------------------------------------


def test_gemini_client_returned_when_configured(gemini_settings):
    client = llm.get_llm_client()

    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-2.5-flash"
    assert client.api_key == "fake-key"
    assert client.base_url.endswith("generativelanguage.googleapis.com/v1beta")


def test_gemini_respects_base_url_override(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "llm_model", "gemini-2.5-flash")
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    monkeypatch.setattr(llm.settings, "llm_base_url", "http://gemini-proxy:8080/")

    client = llm.get_llm_client()

    assert client.base_url == "http://gemini-proxy:8080"


def test_gemini_missing_model_raises_config_error(gemini_settings, monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_model", "")

    with pytest.raises(LLMConfigError):
        llm.get_llm_client()


def test_gemini_missing_api_key_raises_config_error(gemini_settings, monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_api_key", "")

    with pytest.raises(LLMConfigError):
        llm.get_llm_client()


def test_existing_providers_still_selected(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm.settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    monkeypatch.setattr(llm.settings, "llm_base_url", "")
    llm._client = None
    assert isinstance(llm.get_llm_client(), OpenAICompatibleClient)

    monkeypatch.setattr(llm.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(llm.settings, "llm_model", "claude-3-5-haiku-latest")
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    llm._client = None
    assert isinstance(llm.get_llm_client(), AnthropicClient)

    monkeypatch.setattr(llm.settings, "llm_provider", "ollama")
    monkeypatch.setattr(llm.settings, "llm_model", "llama3")
    monkeypatch.setattr(llm.settings, "llm_api_key", "")
    llm._client = None
    assert isinstance(llm.get_llm_client(), OpenAICompatibleClient)


# --- Gemini generation (mocked httpx) --------------------------------------


def test_gemini_complete_sends_prompt_and_returns_text(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "  Gemini answer.  "}]}}]}
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = GeminiClient(
        model="gemini-2.5-flash", api_key="secret-key", base_url="https://example.com/v1beta"
    )

    answer = client.complete("Hello", max_tokens=128)

    assert answer == "Gemini answer."
    assert captured["url"] == "https://example.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "Hello"
    assert captured["json"]["generationConfig"]["maxOutputTokens"] == 128
    # The key travels in the header, never in the URL or body.
    assert captured["headers"]["x-goog-api-key"] == "secret-key"
    assert "secret-key" not in captured["url"]
    assert "secret-key" not in str(captured["json"])


def test_gemini_complete_http_error_raises_llm_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = GeminiClient(model="m", api_key="k", base_url="https://example.com")

    with pytest.raises(LLMError):
        client.complete("hello")


def test_gemini_non_2xx_surfaces_sanitized_google_error(monkeypatch):
    """A 401 must expose Google's error message/status, never the key/headers."""
    google_body = {
        "error": {
            "code": 401,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "UNAUTHENTICATED",
        }
    }
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _RaisingResponse(401, google_body)

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = GeminiClient(
        model="m", api_key="secret-key", base_url="https://example.com"
    )

    with pytest.raises(LLMError) as excinfo:
        client.complete("hello")

    msg = str(excinfo.value)
    assert "401" in msg
    assert "UNAUTHENTICATED" in msg
    assert "API key not valid. Please pass a valid API key." in msg
    # Sanitized: Google's fields only -- never the credential or headers.
    assert "secret-key" not in msg
    assert "x-goog-api-key" not in msg


def test_gemini_complete_unexpected_response_raises_llm_error(monkeypatch):
    def fake_post(*args, **kwargs):
        return _FakeResponse({"unexpected": "shape"})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = GeminiClient(model="m", api_key="k", base_url="https://example.com")

    with pytest.raises(LLMError):
        client.complete("hello")
