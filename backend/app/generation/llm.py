"""Small, provider-agnostic LLM client.

The RAG pipeline only ever talks to an ``LLMClient``; which concrete client is
used is decided here from environment configuration (``LLM_PROVIDER``). Clients
are shipped with no extra dependencies (httpx is already pinned):

- ``OpenAICompatibleClient`` — OpenAI and Ollama both speak the OpenAI
  ``/chat/completions`` protocol.
- ``AnthropicClient`` — the Anthropic ``/messages`` protocol.
- ``GeminiClient`` — the Google Gemini ``generateContent`` REST protocol.

Tests swap the cached ``_client`` for a fake, mirroring how the embedder allows
tests to install a fake model.

Baseline only: a single call, no retries (retries belong to later phases).
"""

from __future__ import annotations

from typing import Protocol

import httpx

from app.core.config import settings
from app.core.exceptions import LLMConfigError, LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Cached client instance, shared across requests. Tests may set this to a fake
# to avoid any real provider call.
_client: "LLMClient | None" = None

_REQUEST_TIMEOUT_SECONDS = 60.0

_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class LLMClient(Protocol):
    """Generates a completion for a single prompt."""

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Return the text completion for ``prompt``. Raises ``LLMError``."""
        ...


class OpenAICompatibleClient:
    """Client for the OpenAI chat-completions protocol (also used by Ollama)."""

    def __init__(self, *, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {exc}") from exc


class AnthropicClient:
    """Client for the Anthropic Messages API."""

    def __init__(self, *, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        try:
            response = httpx.post(
                f"{_ANTHROPIC_BASE_URL}/v1/messages",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        data = response.json()
        try:
            return data["content"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {exc}") from exc


def _gemini_error_message(exc: httpx.HTTPStatusError) -> str:
    """Build a sanitized ``LLMError`` message from a Gemini HTTP error response.

    Surfaces only Google's error body (HTTP status, error reason/status and
    error message). The API key, request headers and any other credential
    material are never included.
    """
    status = exc.response.status_code
    try:
        body = exc.response.json()
        error = body.get("error") if isinstance(body, dict) else None
    except (ValueError, AttributeError, TypeError):
        error = None

    if isinstance(error, dict):
        message = error.get("message")
        reason = error.get("status")
        if message:
            reason_part = f", reason={reason}" if reason else ""
            return f"Google Gemini API error (HTTP {status}{reason_part}): {message}"
    return f"Google Gemini API error (HTTP {status})"


class GeminiClient:
    """Client for the Google Gemini REST API (``generateContent``).

    The full baseline prompt is sent as a single user turn. The API key is sent
    in the ``x-goog-api-key`` header so it never appears in the request URL.
    """

    def __init__(self, *, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        url = f"{self.base_url}/models/{self.model}:generateContent"
        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(_gemini_error_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {exc}") from exc


def _build_client_from_settings() -> LLMClient:
    """Construct the concrete client dictated by ``settings.llm_provider``."""
    provider = (settings.llm_provider or "").strip().lower()

    if provider == "openai":
        if not settings.llm_model:
            raise LLMConfigError("LLM_MODEL is required for provider 'openai'")
        if not settings.llm_api_key:
            raise LLMConfigError("LLM_API_KEY is required for provider 'openai'")
        return OpenAICompatibleClient(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or _OPENAI_DEFAULT_BASE_URL,
        )

    if provider == "ollama":
        if not settings.llm_model:
            raise LLMConfigError("LLM_MODEL is required for provider 'ollama'")
        return OpenAICompatibleClient(
            model=settings.llm_model,
            api_key="",
            base_url=settings.llm_base_url or _OLLAMA_DEFAULT_BASE_URL,
        )

    if provider == "anthropic":
        if not settings.llm_model:
            raise LLMConfigError("LLM_MODEL is required for provider 'anthropic'")
        if not settings.llm_api_key:
            raise LLMConfigError("LLM_API_KEY is required for provider 'anthropic'")
        return AnthropicClient(model=settings.llm_model, api_key=settings.llm_api_key)

    if provider == "gemini":
        if not settings.llm_model:
            raise LLMConfigError("LLM_MODEL is required for provider 'gemini'")
        if not settings.llm_api_key:
            raise LLMConfigError("LLM_API_KEY is required for provider 'gemini'")
        return GeminiClient(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or _GEMINI_DEFAULT_BASE_URL,
        )

    if not provider:
        raise LLMConfigError(
            "LLM_PROVIDER is not configured. Set LLM_PROVIDER (e.g. openai, "
            "anthropic, ollama) in backend/.env before querying."
        )

    raise LLMConfigError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def get_llm_client() -> LLMClient:
    """Return the cached LLM client, building it from settings on first use.

    Tests may set ``llm._client`` to a fake to bypass provider configuration.
    """
    global _client
    if _client is None:
        _client = _build_client_from_settings()
        logger.info("llm client created provider=%s", settings.llm_provider)
    return _client
