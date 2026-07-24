from src.chunking import build_chunks, chunk_text


def test_chunk_text_respects_size_roughly():
    text = "Sentence one. " * 50
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)  # small slack for split-on-boundary


def test_chunk_text_overlap_shares_context():
    text = "AAAA BBBB CCCC DDDD " * 20
    no_overlap = chunk_text(text, chunk_size=40, overlap=0)
    with_overlap = chunk_text(text, chunk_size=40, overlap=10)
    assert len(with_overlap) >= len(no_overlap)


def test_build_chunks_reads_sample_documents():
    chunks = build_chunks("documents", chunk_size=300, overlap=40)
    assert len(chunks) > 0
    sources = {c.source for c in chunks}
    assert "langgraph_overview.txt" in sources
    assert "crewai_overview.txt" in sources
    assert "rag_overview.txt" in sources


def test_chunk_ids_are_unique():
    chunks = build_chunks("documents", chunk_size=300, overlap=40)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
