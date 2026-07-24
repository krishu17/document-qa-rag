from src.chunking import build_chunks
from src.embeddings import MockEmbedder
from src.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from src.keyword_search import KeywordIndex
from src.vectorstore import VectorStore


def _build_retriever():
    embedder = MockEmbedder()
    chunks = build_chunks("documents", chunk_size=300, overlap=40)
    store = VectorStore(dim=embedder.dim)
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    kw = KeywordIndex(chunks)
    return HybridRetriever(store, kw, embedder)


def test_rrf_ranks_items_in_both_lists_highest():
    from src.chunking import Chunk

    a = Chunk(text="a", source="a.txt", chunk_index=0)
    b = Chunk(text="b", source="b.txt", chunk_index=0)
    c = Chunk(text="c", source="c.txt", chunk_index=0)
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    # b is #2 in list1 and #1 in list2 -> should outrank a and c
    assert fused[0][0].source == "b.txt"


def test_hybrid_retriever_returns_relevant_chunks():
    retriever = _build_retriever()
    results = retriever.retrieve("How does CrewAI structure its agents?", top_k=3)
    sources = [c.source for c, _ in results]
    assert "crewai_overview.txt" in sources


def test_hybrid_retriever_respects_top_k():
    retriever = _build_retriever()
    results = retriever.retrieve("LangGraph", top_k=2)
    assert len(results) <= 2
