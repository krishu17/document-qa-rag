"""Embedding provider abstraction.

Selected via EMBEDDING_PROVIDER:

    openai  -> text-embedding-3-small via the OpenAI API (needs OPENAI_API_KEY)
    mock    -> deterministic offline stand-in (default)

The mock embedder is NOT semantic -- it hashes overlapping word shingles into
a fixed-size vector (a simplified feature-hashing / "hashing trick" bag-of-words
embedding). It gives similar text similar vectors for exact/near-exact word
overlap, which is enough to exercise the vector store, hybrid search, and
end-to-end pipeline in tests without any API key or network access. It is not
a substitute for a real semantic embedding model -- swap EMBEDDING_PROVIDER to
`openai` for real semantic retrieval quality.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List

import numpy as np

MOCK_DIM = 256


def _hash_embed(text: str, dim: int = MOCK_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    words = re.findall(r"[a-z0-9]+", text.lower())
    shingles = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    if not shingles:
        return vec
    for token in shingles:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Embedder:
    def embed(self, texts: List[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class MockEmbedder(Embedder):
    def __init__(self, dim: int = MOCK_DIM):
        self._dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.stack([_hash_embed(t, self._dim) for t in texts])

    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._dim = 1536 if model == "text-embedding-3-small" else 3072

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [d.embedding for d in response.data]
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    @property
    def dim(self) -> int:
        return self._dim


def get_embedder() -> Embedder:
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").lower()
    if provider == "openai":
        return OpenAIEmbedder(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    if provider == "mock":
        return MockEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")
