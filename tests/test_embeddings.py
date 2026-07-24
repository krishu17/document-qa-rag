from src.embeddings import MockEmbedder


def test_mock_embedder_returns_unit_vectors():
    embedder = MockEmbedder(dim=64)
    vecs = embedder.embed(["hello world", "another sentence"])
    assert vecs.shape == (2, 64)
    for v in vecs:
        norm = (v**2).sum() ** 0.5
        assert abs(norm - 1.0) < 1e-4 or norm == 0.0


def test_mock_embedder_similar_text_scores_higher():
    embedder = MockEmbedder()
    a, b, c = embedder.embed([
        "LangGraph builds multi-agent workflows with graphs.",
        "LangGraph is used to build agent workflows using graphs.",
        "Bananas are a good source of potassium.",
    ])
    assert float(a @ b) > float(a @ c)
