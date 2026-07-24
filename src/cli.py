"""Command-line entry point.

    python -m src.cli "your question" [--top-k 4] [--docs documents]
"""
from __future__ import annotations

import argparse

from .rag import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the document Q&A RAG pipeline.")
    parser.add_argument("question", help="Question to ask about the ingested documents")
    parser.add_argument("--docs", default="documents", help="Directory of .txt documents to ingest")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    pipeline = RagPipeline.from_directory(args.docs)
    result = pipeline.answer(args.question, top_k=args.top_k)

    print("Question:", result.question)
    print()
    print("Answer:", result.answer)
    print()
    print("Retrieved chunks:")
    for chunk, score in result.sources:
        print(f"  [{score:.3f}] {chunk.id}")


if __name__ == "__main__":
    main()
