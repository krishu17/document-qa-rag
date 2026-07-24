"""Document loading and chunking.

Loads plain-text documents from a directory and splits them into overlapping
chunks using a recursive character splitter: try to break on paragraph
boundaries first, then sentences, then words, only falling back to a hard
character cut if nothing else fits within chunk_size. This keeps chunks from
splitting mid-sentence when the source text allows it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int

    @property
    def id(self) -> str:
        return f"{self.source}::{self.chunk_index}"


def load_documents(directory: str) -> List[tuple[str, str]]:
    """Return a list of (filename, text) for every .txt file in directory."""
    docs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(directory, fname)
        with open(path, "r", encoding="utf-8") as f:
            docs.append((fname, f.read()))
    return docs


def _split_text(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size or not separators:
        return [text] if text.strip() else []

    sep, rest = separators[0], separators[1:]
    if sep == "":
        # Hard cut -- last resort.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, rest))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    """Split text into chunks of at most chunk_size characters with overlap.

    Overlap is applied as a simple sliding window over the recursively split
    pieces so consecutive chunks share some context, which helps retrieval
    when a relevant sentence sits right at a chunk boundary.
    """
    raw_chunks = _split_text(text.strip(), chunk_size, _SEPARATORS)
    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        prev_tail = raw_chunks[i - 1][-overlap:]
        overlapped.append((prev_tail + " " + chunk).strip())
    return overlapped


def build_chunks(directory: str, chunk_size: int = 400, overlap: int = 60) -> List[Chunk]:
    chunks: List[Chunk] = []
    for fname, text in load_documents(directory):
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(text=piece, source=fname, chunk_index=i))
    return chunks
