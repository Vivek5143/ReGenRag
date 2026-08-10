"""Embedding generation with a local sentence-transformers model.

The configured ``EMBEDDING_MODEL`` runs fully on-device: no API key, no
external calls, no uploads. The model is loaded lazily once and cached for the
process. At load time the model's actual output dimension is compared against
``EMBEDDING_DIMENSION`` and ingestion refuses to run on a mismatch, so we never
silently store a wrong-sized vector. ``embed_texts`` supports batch encoding.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Cached model instance, shared across requests. Tests may swap this with a
# lightweight fake to avoid downloading a real model.
_model = None


def _load_model():
    """Load and validate the configured embedding model (called once)."""
    model_name = settings.embedding_model
    if not model_name:
        raise DocumentProcessingError(
            "EMBEDDING_MODEL is not configured. Set it in backend/.env."
        )
    try:
        # Imported here so importing the embedder never pulls in torch.
        from sentence_transformers import SentenceTransformer

        import torch

        # Seed at load time so embeddings are deterministic for a given input.
        torch.manual_seed(settings.embedding_seed)
        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise DocumentProcessingError(
            f"Failed to load embedding model '{model_name}': {exc}"
        ) from exc

    dimension = model.get_sentence_embedding_dimension()
    if dimension != settings.embedding_dimension:
        raise DocumentProcessingError(
            f"Embedding model '{model_name}' outputs {dimension} dimensions but "
            f"EMBEDDING_DIMENSION={settings.embedding_dimension}. Align the "
            "model and configuration before ingesting documents."
        )

    logger.info(
        "embedding model loaded model=%s dimension=%s", model_name, dimension
    )
    return model


def get_embedding_model():
    """Return the cached embedding model, loading it on first use."""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode ``texts`` into a list of normalized embedding vectors (batched)."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Alias kept for the ingestion pipeline; delegates to ``embed_texts``."""
    return embed_texts(texts)