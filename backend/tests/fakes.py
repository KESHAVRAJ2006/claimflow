"""Test doubles shared across test modules."""

import hashlib
import math
import re
from collections.abc import Sequence

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic bag-of-words embedder: fast, offline, and similar texts get similar vectors.

    Good enough to test ingestion, filtering and ranking plumbing; retrieval quality is measured with the real
    model in test_retrieval_eval.py.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self._dimension
            for word in _WORD.findall(text.lower()):
                bucket = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % self._dimension
                vector[bucket] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors
