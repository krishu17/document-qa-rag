"""BM25 keyword search over the same chunk set used by the vector store.

Kept separate from VectorStore so the two retrieval signals are computed
independently and then combined in hybrid_search.py -- semantic and keyword
search tend to fail on different query types (BM25 is strong on exact terms
like error codes or proper nouns; embeddings are strong on paraphrase), so
keeping them decoupled makes it possible to combine or A/B them.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .chunking import Chunk


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordIndex:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = list(chunks)
        tokenized = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        query_tokens = set(_tokenize(query))
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            # BM25's IDF term can land on exactly 0 for a query term that
            # appears in roughly half of a *very* small corpus (its formula
            # is log((N - n + 0.5) / (n + 0.5)), which is 0 when n == N/2),
            # so a real match can still score 0. Score sign alone can't
            # distinguish that from "no match" -- require actual token
            # overlap with the document as well.
            doc_tokens = set(_tokenize(self.chunks[idx].text))
            if scores[idx] < 0 or not (query_tokens & doc_tokens):
                continue
            results.append((self.chunks[idx], float(scores[idx])))
        return results

    def __len__(self) -> int:
        return len(self.chunks)
