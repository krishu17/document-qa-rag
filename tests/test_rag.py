import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "mock"

from src.rag import RagPipeline  # noqa: E402


def _pipeline():
    return RagPipeline.from_directory("documents", chunk_size=300, overlap=40)


def test_answer_grounded_question_includes_citation():
    pipeline = _pipeline()
    result = pipeline.answer("What does the Send API in LangGraph do?", top_k=3)
    assert "[" in result.answer and "]" in result.answer
    assert result.sources


def test_answer_out_of_scope_question_declines():
    pipeline = _pipeline()
    result = pipeline.answer("What is the capital of France?", top_k=3)
    assert "don't have enough information" in result.answer.lower()


def test_empty_pipeline_has_no_sources():
    pipeline = RagPipeline()
    result = pipeline.answer("anything", top_k=3)
    assert result.sources == []
    assert "don't have enough information" in result.answer.lower()
