from src.chunking import Chunk
from src.keyword_search import KeywordIndex


def test_bm25_finds_exact_term_match():
    chunks = [
        Chunk(text="Reciprocal Rank Fusion merges two ranked lists.", source="a.txt", chunk_index=0),
        Chunk(text="Bananas are a good source of potassium.", source="b.txt", chunk_index=0),
    ]
    idx = KeywordIndex(chunks)
    results = idx.search("Reciprocal Rank Fusion", top_k=2)
    assert results[0][0].source == "a.txt"


def test_bm25_no_match_returns_empty():
    chunks = [Chunk(text="Bananas are tasty.", source="a.txt", chunk_index=0)]
    idx = KeywordIndex(chunks)
    assert idx.search("quantum entanglement", top_k=2) == []


def test_empty_index():
    idx = KeywordIndex([])
    assert idx.search("anything", top_k=2) == []
