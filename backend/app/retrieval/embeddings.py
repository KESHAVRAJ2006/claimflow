"""Sentence embeddings for retrieval.

The model loads once per process: ``get_embedder`` is cached, and the FastAPI lifespan calls it at startup.
Loading all-MiniLM-L6-v2 takes seconds and ~100 MB of memory, so doing it per request would be ruinous.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that turns texts into unit-length vectors. Tests substitute a fast fake."""

    @property
    def dimension(self) -> int:
        """Length of each vector."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts into L2-normalised vectors, in input order."""
        ...


class SentenceTransformerEmbedder:
    """Embedder backed by a local sentence-transformers model, on CPU."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load the model.

        Args:
            model_name: Hugging Face model ID. In Docker the model is baked into the image at build time.
        """
        # Imported here, not at module top, because importing sentence_transformers pulls in torch (seconds);
        # modules that only need the Embedder protocol, and most tests, stay fast.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self._dimension = int(self._model.get_embedding_dimension())

    @property
    def dimension(self) -> int:
        """Length of each vector (384 for all-MiniLM-L6-v2)."""
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts.

        Args:
            texts: Texts to embed.

        Returns:
            One normalised vector per text. Normalising makes cosine similarity equal the dot product.
        """
        vectors = self._model.encode(
            list(texts), batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.tolist()


@lru_cache(maxsize=2)
def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformerEmbedder:
    """Return the process-wide embedder for a model, loading it on first call only.

    Args:
        model_name: Hugging Face model ID.

    Returns:
        The cached SentenceTransformerEmbedder.
    """
    return SentenceTransformerEmbedder(model_name)
