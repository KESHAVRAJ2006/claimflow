"""The production embedder (ONNX, no torch) must produce the same vectors as the reference (sentence-transformers).

Qdrant is filled and queried by whichever embedder the process has, so the two must agree: otherwise retrieval
quality measured in development (scripts.eval_retrieval) would say nothing about production.
"""

import importlib.util
import math
from collections.abc import Iterator

import pytest

from app.retrieval import embeddings
from app.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, OnnxEmbedder, SentenceTransformerEmbedder, get_embedder

TEXTS = [
    "Is theft from an unlocked car covered?",
    "Cataract surgery is covered only after 24 months of continuous coverage, whatever its cause.",
    "x",
    # Longer than the model's 256-token limit, so truncation must match too.
    "Damage from water leaking or seeping over a period of time is excluded. " * 40,
]


@pytest.fixture(scope="module")
def reference() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)


@pytest.fixture(scope="module")
def onnx() -> OnnxEmbedder:
    return OnnxEmbedder(DEFAULT_EMBEDDING_MODEL)


@pytest.fixture
def fresh_cache() -> Iterator[None]:
    get_embedder.cache_clear()
    yield
    get_embedder.cache_clear()


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # both are unit length


def test_onnx_vectors_match_sentence_transformers(reference: SentenceTransformerEmbedder, onnx: OnnxEmbedder) -> None:
    assert onnx.dimension == reference.dimension == 384
    for expected, actual in zip(reference.embed(TEXTS), onnx.embed(TEXTS), strict=True):
        assert math.isclose(math.fsum(v * v for v in actual), 1.0, abs_tol=1e-5)
        assert _cosine(expected, actual) > 0.99999
        assert max(abs(e - a) for e, a in zip(expected, actual, strict=True)) < 1e-4


def test_onnx_batches_give_the_same_vectors_as_single_texts(onnx: OnnxEmbedder) -> None:
    # Padding in a batch must not leak into the mean: each text alone gives the same vector.
    batch = onnx.embed(TEXTS)
    for text, vector in zip(TEXTS, batch, strict=True):
        assert _cosine(onnx.embed([text])[0], vector) > 0.99999
    many = [f"claim number {n}" for n in range(70)]  # more than one batch of 32
    assert len(onnx.embed(many)) == 70 and onnx.embed([]) == []


def test_production_uses_onnx_when_sentence_transformers_is_absent(
    monkeypatch: pytest.MonkeyPatch, fresh_cache: None
) -> None:
    assert isinstance(get_embedder(DEFAULT_EMBEDDING_MODEL), SentenceTransformerEmbedder)  # development image
    get_embedder.cache_clear()
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        embeddings.importlib.util,
        "find_spec",
        lambda name, *args: None if name == "sentence_transformers" else real_find_spec(name, *args),
    )
    assert isinstance(get_embedder(DEFAULT_EMBEDDING_MODEL), OnnxEmbedder)  # production image
