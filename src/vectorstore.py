"""Thin wrapper around a FAISS inner-product index for semantic search.

Embeddings are expected to already be L2-normalized (see embeddings.py), so
inner product search is equivalent to cosine similarity search. Chunk
metadata (source filename, chunk index, text) is kept in a parallel Python
list indexed by FAISS's internal integer ids.
"""
from __future__ import annotations

from typing import List, Tuple

import faiss
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {vectors.shape[1]}")
        self.index.add(vectors.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def __len__(self) -> int:
        return len(self.chunks)
