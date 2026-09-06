"""Local embedding model for RAG. No paid/cloud embedding APIs are used.

The model name is fully configurable via the EMBEDDING_MODEL environment
variable, so operators can swap in any sentence-transformers compatible
model that fits their hardware.
"""
from __future__ import annotations

import threading

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the local embedding model cannot be loaded (e.g. not downloaded, no network)."""


class LocalEmbeddingFunction:
    """Chroma-compatible embedding function backed by sentence-transformers.

    Implements the `__call__(self, input: list[str]) -> list[list[float]]` protocol
    that chromadb.api.types.EmbeddingFunction expects.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("embedding_model_loading", extra={"extra_fields": {"model": self.model_name}})
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    f"Could not load local embedding model '{self.model_name}'. "
                    f"If this is the first run, the model needs to download from Hugging Face "
                    f"the first time (requires network access) or be pre-cached. "
                    f"Set EMBEDDING_MODEL in .env to a different local model if needed. "
                    f"Original error: {exc}"
                ) from exc
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - chroma protocol name
        model = self._load()
        embeddings = model.encode(list(input), convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def name(self) -> str:
        return f"local:{self.model_name}"


_embedding_fn: LocalEmbeddingFunction | None = None


def get_embedding_function() -> LocalEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = LocalEmbeddingFunction()
    return _embedding_fn
