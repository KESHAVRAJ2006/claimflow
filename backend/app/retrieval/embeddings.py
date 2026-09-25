"""Sentence embeddings for retrieval.

The model loads once per process: ``get_embedder`` is cached, and the FastAPI lifespan calls it at startup.
Loading all-MiniLM-L6-v2 takes seconds and ~100 MB of memory, so doing it per request would be ruinous.

Two implementations of the same model:
- ``SentenceTransformerEmbedder``, the reference, used in development, tests and evaluation.
- ``OnnxEmbedder``, the model's ONNX export run without torch, used by the production image. Importing
  sentence-transformers pulls in torch: about 400 MB before the model even loads, which alone would fill a
  512 MB free instance. Both produce the same vectors (``tests/test_embeddings.py`` compares them).
"""

import importlib.util
import json
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The files OnnxEmbedder reads from the model's repository; the Dockerfile downloads exactly these at build time.
ONNX_MODEL_FILES = ("onnx/model.onnx", "tokenizer.json", "sentence_bert_config.json", "1_Pooling/config.json")
BATCH_SIZE = 32


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
            list(texts), batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )  # fmt: skip
        return vectors.tolist()


class OnnxEmbedder:
    """The same model's ONNX export, run with onnxruntime: no torch, about a third of the memory.

    Reproduces sentence-transformers' pipeline for a mean-pooling model: tokenise (truncated to the model's
    max_seq_length), run the transformer, average the token vectors over real (unpadded) tokens, L2-normalise.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load the ONNX model and its tokenizer.

        Args:
            model_name: Hugging Face model ID; its repository must include an ONNX export (``onnx/model.onnx``).

        Raises:
            ValueError: If the model does not use mean pooling, the only pooling reproduced here.
        """
        import numpy as np
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        paths = {name: hf_hub_download(model_name, name) for name in ONNX_MODEL_FILES}
        if not _read_json(paths["1_Pooling/config.json"]).get("pooling_mode_mean_tokens"):
            raise ValueError(f"{model_name} does not use mean pooling; use SentenceTransformerEmbedder")
        max_tokens = int(_read_json(paths["sentence_bert_config.json"])["max_seq_length"])

        self.model_name = model_name
        self._np = np
        self._tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        self._tokenizer.enable_truncation(max_tokens)
        self._tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        # One thread: free instances have a fraction of a CPU, and more threads only add memory and contention.
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            paths["onnx/model.onnx"], options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {model_input.name for model_input in self._session.get_inputs()}
        self._dimension = len(self.embed(["dimension probe"])[0])

    @property
    def dimension(self) -> int:
        """Length of each vector (384 for all-MiniLM-L6-v2)."""
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts.

        Args:
            texts: Texts to embed.

        Returns:
            One normalised vector per text, as sentence-transformers would produce.
        """
        np = self._np
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            encodings = self._tokenizer.encode_batch(list(texts[start : start + BATCH_SIZE]))
            ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
            mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            tokens = self._session.run(None, feed)[0]  # (batch, sequence, dimension)
            weights = mask[..., None].astype(tokens.dtype)
            pooled = (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            vectors.extend(pooled.tolist())
        return vectors


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)
    return data


@lru_cache(maxsize=2)
def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    """Return the process-wide embedder for a model, loading it on first call only.

    sentence-transformers where it is installed (development, tests, evaluation); otherwise the ONNX export
    (the production image, which leaves torch out to fit a 512 MB instance).

    Args:
        model_name: Hugging Face model ID.

    Returns:
        The cached embedder.
    """
    if importlib.util.find_spec("sentence_transformers") is not None:
        return SentenceTransformerEmbedder(model_name)
    return OnnxEmbedder(model_name)
