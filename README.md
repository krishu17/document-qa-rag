# Document Q&A Assistant (RAG)

A retrieval-augmented generation pipeline over a small local document set:
ingestion → chunking → hybrid retrieval (vector + keyword) → re-rank →
grounded generation with citations back to the source chunk.

```
documents/*.txt
      │  chunk_text() -- recursive splitter, paragraph/sentence/word fallback
      ▼
   Chunk[]
      │  embed()                         │  BM25Okapi()
      ▼                                  ▼
 FAISS IndexFlatIP                 KeywordIndex
      │  vector search (top N)           │  keyword search (top N)
      └──────────────┬───────────────────┘
                      ▼
        Reciprocal Rank Fusion (rank-based merge)
                      ▼
          TF-IDF cosine re-rank (lexical second pass)
                      ▼
                top_k chunks ──► prompt ──► LLM ──► answer + [citation]
```

## Why hybrid search + a separate re-rank pass

Vector search is strong on paraphrase ("how do I query a database in
English?" → chunks about a "natural-language SQL agent") but can miss exact
terms. BM25 is strong on exact terms (error codes, proper nouns, acronyms
like "RRF") but blind to paraphrase. Reciprocal Rank Fusion merges the two
*ranked lists* by position rather than trying to normalize cosine similarity
and BM25 scores onto the same scale, which don't mean the same thing. The
TF-IDF re-rank pass afterward is a second, independent lexical signal over
just the fused candidates — a real (if lightweight) re-ranking stage, not a
neural cross-encoder, and the code says so.

## Pluggable embeddings and LLM

Both are selected via environment variables so the pipeline runs fully
offline by default and can be pointed at a real model with no code changes:

| | `mock` (default) | real |
|---|---|---|
| `EMBEDDING_PROVIDER` | hashing-trick bag-of-words vectors (`src/embeddings.py`) | `openai` → `text-embedding-3-small` |
| `LLM_PROVIDER` | extractive stand-in (`src/mock_llm.py`) | `openai` / `anthropic` |

The mock embedder is **not semantic** — it hashes overlapping word shingles
into a fixed-size vector, so it rewards exact/near-exact word overlap rather
than meaning. That's enough to exercise chunking, indexing, hybrid search,
and the "don't answer outside the context" guardrail end-to-end in tests
with zero API calls and zero network access. It is not a claim that offline
testing is equivalent to testing real semantic retrieval quality — swap in
`openai` embeddings for that.

The mock LLM is extractive, not generative: it picks the retrieved chunk
with the highest (stopword-filtered) word overlap with the question and
returns it verbatim with a citation, or declines if nothing overlaps. That
last part is what actually exercises the "don't hallucinate beyond the
context" behavior in `tests/test_rag.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # or leave both providers on `mock`
```

## Run it

```bash
# Fully offline
python -m src.cli "What does the Send API in LangGraph do?"

# With real embeddings + generation
EMBEDDING_PROVIDER=openai LLM_PROVIDER=openai OPENAI_API_KEY=sk-... \
  python -m src.cli "What does the Send API in LangGraph do?"
```

Drop more `.txt` files into `documents/` to expand the corpus — no other
changes needed.

## Test

```bash
pytest -q
```

18 tests, all exercising real code paths (no network, no API key):
chunking (size limits, overlap, unique chunk ids), the mock embedder
(unit-norm output, similar text scores higher than unrelated text), the
FAISS vector store (correct nearest neighbor, dimension mismatch raises,
empty store), BM25 keyword search (including a real bug I hit and fixed —
see below), Reciprocal Rank Fusion, and the full pipeline (grounded answers
carry a citation, out-of-scope questions get declined, an empty pipeline
degrades safely).

## A real bug worth knowing for an interview

`test_bm25_finds_exact_term_match` initially failed with an `IndexError`
because BM25's IDF term, `log((N - n + 0.5) / (n + 0.5))`, evaluates to
**exactly 0** when a query term appears in precisely half the documents in a
very small corpus (2 documents, 1 containing the term). The original filter
(`if scores[idx] <= 0: continue`) silently dropped a real match. Fixing it
required requiring actual token overlap between the query and the document
as a second condition, rather than relying on BM25's score sign alone to
mean "no match" — score sign and "relevance" aren't the same thing at small
corpus sizes. See the comment in `src/keyword_search.py`.

## Docker

```bash
docker build -t document-qa-rag .
docker run document-qa-rag "What does the Send API in LangGraph do?"
```

## Known limitations

- The mock embedder and mock LLM are offline stand-ins for exercising
  control flow, not a claim of real retrieval or generation quality.
- Chunking is a simple recursive character splitter (paragraph → sentence →
  word → hard cut), not a token-aware or semantic chunker.
- The TF-IDF re-rank is a lexical signal, not a neural cross-encoder.
- BM25 on very small corpora can produce degenerate (zero) IDF values for
  common terms — see the bug note above.
