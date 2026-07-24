from src.chunking import Chunk
from src.embeddings import MockEmbedder
from src.vectorstore import VectorStore


def _chunks():
    return [
        Chunk(text="LangGraph supports parallel fan-out via Send.", source="a.txt", chunk_index=0),
        Chunk(text="CrewAI uses role-based agents in a sequential process.", source="b.txt", chunk_index=0),
    ]


def test_vector_store_returns_most_similar_chunk_first():
    embedder = MockEmbedder()
    chunks = _chunks()
    store = VectorStore(dim=embedder.dim)
    store.add(chunks, embedder.embed([c.text for c in chunks]))

    query_vec = embedder.embed(["What does the Send API do in LangGraph?"])[0]
    results = store.search(query_vec, top_k=2)
    assert results[0][0].source == "a.txt"


def test_vector_store_rejects_wrong_dim():
    embedder = MockEmbedder(dim=32)
    store = VectorStore(dim=64)
    import pytest

    with pytest.raises(ValueError):
        store.add(_chunks(), embedder.embed([c.text for c in _chunks()]))


def test_empty_store_search_returns_empty():
    store = VectorStore(dim=16)
    import numpy as np

    assert store.search(np.zeros(16), top_k=3) == []
