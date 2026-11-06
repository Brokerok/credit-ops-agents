"""Embeddings. Default: a small multilingual ONNX model via fastembed (runs on CPU, no API key, handles
Ukrainian/Russian/English). `hash` is a dependency-free bag-of-words embedder for unit tests and CI."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"\w+", re.U)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, model-free. Good enough to test retrieval plumbing and role filtering."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class FastEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = int(next(iter(self._model.embed(["dim probe"]))).shape[0])

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> Embedder:
    return HashEmbedder() if model_name == "hash" else FastEmbedder(model_name)
