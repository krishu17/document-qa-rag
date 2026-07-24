"""End-to-end RAG pipeline: ingest -> hybrid retrieve -> generate with citations.

    RagPipeline.from_directory("documents") builds the chunk set, embeds it,
    and indexes it into both the vector store and the BM25 index.
    .answer(question) retrieves the top chunks via HybridRetriever and asks
    the configured LLM (see llm.py) to answer using only that context,
    returning both the answer text and the chunks it was grounded in so the
    caller can show citations / let the user verify the claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.messages import HumanMessage

from .chunking import Chunk, build_chunks
from .embeddings import Embedder, get_embedder
from .hybrid_search import HybridRetriever
from .keyword_search import KeywordIndex
from .llm import get_llm
from .vectorstore import VectorStore


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: List[Tuple[Chunk, float]]


class RagPipeline:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.vector_store = VectorStore(dim=self.embedder.dim)
        self.keyword_index: KeywordIndex | None = None
        self.retriever: HybridRetriever | None = None

    @classmethod
    def from_directory(cls, directory: str, chunk_size: int = 400, overlap: int = 60) -> "RagPipeline":
        pipeline = cls()
        chunks = build_chunks(directory, chunk_size=chunk_size, overlap=overlap)
        pipeline.index(chunks)
        return pipeline

    def index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            self.keyword_index = KeywordIndex([])
            self.retriever = HybridRetriever(self.vector_store, self.keyword_index, self.embedder)
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        self.vector_store.add(chunks, vectors)
        self.keyword_index = KeywordIndex(self.vector_store.chunks)
        self.retriever = HybridRetriever(self.vector_store, self.keyword_index, self.embedder)

    def retrieve(self, question: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.retriever is None:
            return []
        return self.retriever.retrieve(question, top_k=top_k)

    def answer(self, question: str, top_k: int = 4) -> RagAnswer:
        sources = self.retrieve(question, top_k=top_k)
        context_block = "\n".join(f"[{chunk.id}] {chunk.text}" for chunk, _ in sources)

        prompt = (
            "### TASK: ANSWER\n"
            "Answer the question using ONLY the context below. If the context does not "
            "contain the answer, say you don't have enough information. Cite the chunk "
            "id(s) you used in square brackets, e.g. [source.txt::2].\n"
            f"QUESTION:\n{question}\n"
            f"CONTEXT:\n{context_block}\n"
        )
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return RagAnswer(question=question, answer=response.content, sources=sources)
