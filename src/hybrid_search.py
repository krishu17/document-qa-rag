"""Combine vector search and BM25 keyword search via Reciprocal Rank Fusion,
then apply a lightweight TF-IDF re-rank pass on the fused candidates.

RRF fuses two ranked lists by rank position rather than raw score, which
avoids having to normalize BM25 scores and cosine similarities onto the same
scale -- a chunk ranked #1 in either list contributes 1/(k+1) to its fused
score, #2 contributes 1/(k+2), and so on. Chunks that rank well in *both*
lists float to the top even if neither list alone would rank them first.

The re-rank pass is a second, independent relevance signal: a TF-IDF cosine
similarity between the query and each fused candidate, computed fresh (not
reused from BM25). It is a lexical re-ranker, not a neural cross-encoder --
documented here so it isn't mistaken for one.
"""
from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk
from .keyword_search import KeywordIndex
from .vectorstore import VectorStore


def reciprocal_rank_fusion(
    ranked_lists: List[List[Chunk]], k: int = 60
) -> List[Tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunk_by_id[chunk.id] = chunk
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(chunk_by_id[cid], score) for cid, score in fused]


def tfidf_rerank(query: str, chunks: List[Chunk]) -> List[Tuple[Chunk, float]]:
    if not chunks:
        return []
    corpus = [c.text for c in chunks] + [query]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    query_vec = matrix[-1]
    doc_vecs = matrix[:-1]
    sims = cosine_similarity(query_vec, doc_vecs)[0]
    ranked = sorted(zip(chunks, sims), key=lambda cs: cs[1], reverse=True)
    return [(c, float(s)) for c, s in ranked]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, keyword_index: KeywordIndex, embedder):
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5, fusion_pool: int = 10) -> List[Tuple[Chunk, float]]:
        query_vec = self.embedder.embed([query])[0]
        vector_hits = [c for c, _ in self.vector_store.search(query_vec, top_k=fusion_pool)]
        keyword_hits = [c for c, _ in self.keyword_index.search(query, top_k=fusion_pool)]

        if not vector_hits and not keyword_hits:
            return []

        fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
        fused_chunks = [c for c, _ in fused][: max(top_k * 2, fusion_pool)]

        reranked = tfidf_rerank(query, fused_chunks)
        return reranked[:top_k]
