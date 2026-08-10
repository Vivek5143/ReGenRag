"""Tests for the embedder.

These never load the real sentence-transformers model or torch. Instead they
stub the modules that the embedder imports so we can exercise dimension
validation and the batching/determinism of ``embed_texts`` with a light fake.
"""

import sys
import types
import zlib

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError
from app.ingestion import embedder


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """Isolate the module-level model cache between tests."""
    embedder._model = None
    yield
    embedder._model = None


def _stub_module(**attrs):
    module = types.ModuleType("stub")
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _stub_embedding_imports(monkeypatch, dimension):
    """Point the embedder's local imports at fakes with the given dimension."""

    class StubSentenceTransformer:
        def __init__(self, name):
            self._dim = dimension

        def get_sentence_embedding_dimension(self):
            return self._dim

    monkeypatch.setattr(settings, "embedding_model", "fake/embedding-model")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        _stub_module(SentenceTransformer=StubSentenceTransformer),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _stub_module(manual_seed=lambda seed: None),
    )


class DeterministicModel:
    """Fake model returning a stable per-text vector (no real model)."""

    def __init__(self, dim=384):
        self.dim = dim
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            rng = np.random.RandomState(zlib.crc32(text.encode()))
            out[i] = rng.rand(self.dim)
        return out


def test_embeds_batch_with_expected_count_and_dimension():
    model = DeterministicModel(dim=384)
    embedder._model = model

    vectors = embedder.embed_texts(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)
    # All texts were passed in a single batched encode call.
    assert len(model.calls) == 1
    assert model.calls[0][0] == ["alpha", "beta", "gamma"]


def test_embed_documents_delegates_to_batch_path():
    model = DeterministicModel(dim=384)
    embedder._model = model

    vectors = embedder.embed_documents(["only one"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_empty_input_yields_no_vectors():
    embedder._model = DeterministicModel(dim=384)

    assert embedder.embed_texts([]) == []


def test_output_is_deterministic_for_same_input():
    embedder._model = DeterministicModel(dim=384)

    first = embedder.embed_texts(["same text", "same text"])
    second = embedder.embed_texts(["same text", "same text"])

    assert first == second


def test_embedding_dimension_matches_config():
    # Guards against a config/schema drift: the column is Vector(EMBEDDING_DIMENSION).
    assert settings.embedding_dimension == 384


def test_load_model_rejects_dimension_mismatch(monkeypatch):
    _stub_embedding_imports(monkeypatch, dimension=768)  # model != configured 384

    with pytest.raises(DocumentProcessingError):
        embedder.get_embedding_model()


def test_load_model_accepts_matching_dimension(monkeypatch):
    _stub_embedding_imports(monkeypatch, dimension=384)

    model = embedder.get_embedding_model()

    assert model.get_sentence_embedding_dimension() == 384


def test_load_model_requires_configured_model(monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "")

    with pytest.raises(DocumentProcessingError):
        embedder.get_embedding_model()