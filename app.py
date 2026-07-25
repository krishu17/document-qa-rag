"""Streamlit UI for the document Q&A RAG pipeline.

Runs the same RagPipeline defined in src/rag.py -- this file adds no new
retrieval or generation logic, it just gives a human a form instead of a
CLI, plus a way to drop new documents into documents/ without touching a
terminal. Provider selection and API keys are session-only: keys typed
here are written to os.environ for this process only, never to disk.

    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os

import streamlit as st

from src.rag import RagPipeline

st.set_page_config(page_title="Document Q&A", page_icon="\U0001F4C4", layout="centered")

_CSS = """
<style>
:root {
    --accent: #1c7f72;
    --accent-soft: #e2f0ec;
}
@media (prefers-color-scheme: dark) {
    :root { --accent: #5fc9b8; --accent-soft: #132824; }
}
.stApp [data-testid="stHeader"] { background: transparent; }
.rag-hero { padding: 4px 0 18px; border-bottom: 1px solid rgba(128,128,128,.25); margin-bottom: 22px; }
.rag-hero h1 { font-size: 1.9rem; margin: 0 0 4px; letter-spacing: -.01em; }
.rag-hero p { opacity: .72; margin: 0; font-size: .95rem; }
.rag-answer { border-left: 3px solid var(--accent); background: var(--accent-soft); padding: 16px 18px;
    border-radius: 0 10px 10px 0; font-size: 1.02rem; line-height: 1.55; white-space: pre-wrap; }
.rag-score { font-family: ui-monospace, Consolas, monospace; font-size: .78rem; opacity: .7; }
.rag-chunkid { font-family: ui-monospace, Consolas, monospace; font-size: .82rem; font-weight: 600; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="rag-hero"><h1>Document Q&amp;A</h1>'
    "<p>Hybrid retrieval (vector + BM25 + TF-IDF re-rank) over your own documents, "
    "with grounded, cited answers.</p></div>",
    unsafe_allow_html=True,
)

DOCS_DIR = "documents"
os.makedirs(DOCS_DIR, exist_ok=True)

with st.sidebar:
    st.subheader("Retrieval")
    embedding_provider = st.selectbox("EMBEDDING_PROVIDER", ["mock", "openai"], index=0)
    os.environ["EMBEDDING_PROVIDER"] = embedding_provider
    if embedding_provider == "mock":
        st.caption("Hashing-trick bag-of-words vectors — no key needed, rewards word overlap, not meaning.")
    else:
        key = st.text_input("OPENAI_API_KEY", type="password", key="embed_key")
        if key:
            os.environ["OPENAI_API_KEY"] = key
        st.caption("Uses text-embedding-3-small for real semantic retrieval.")

    st.subheader("Generation")
    llm_provider = st.selectbox("LLM_PROVIDER", ["mock", "openai", "anthropic"], index=0)
    os.environ["LLM_PROVIDER"] = llm_provider
    if llm_provider == "mock":
        st.caption("Extractive stand-in — picks the best-overlapping chunk verbatim, no key needed.")
    elif llm_provider == "openai":
        key = st.text_input("OPENAI_API_KEY", type="password", key="llm_key_openai")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    else:
        key = st.text_input("ANTHROPIC_API_KEY", type="password", key="llm_key_anthropic")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key

    st.caption("Keys are kept in memory for this session only — never written to disk.")

    st.subheader("Retrieval settings")
    top_k = st.slider("top_k (chunks retrieved)", 1, 10, 4)
    with st.expander("Chunking (advanced)"):
        chunk_size = st.number_input("chunk_size", min_value=100, max_value=2000, value=400, step=50)
        overlap = st.number_input("overlap", min_value=0, max_value=500, value=60, step=10)

    st.subheader("Documents")
    existing = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith((".txt", ".pdf")))
    for fname in existing:
        size_kb = os.path.getsize(os.path.join(DOCS_DIR, fname)) / 1024
        st.caption(f"\U0001F4C4 {fname} ({size_kb:,.0f} KB)")

    uploaded = st.file_uploader("Add a document", type=["txt", "pdf"])
    if uploaded is not None:
        safe_name = os.path.basename(uploaded.name)
        dest = os.path.join(DOCS_DIR, safe_name)
        if st.button(f"Save {safe_name} to documents/"):
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved {safe_name} — it will be indexed on the next question.")
            st.rerun()


@st.cache_resource(show_spinner="Indexing documents…")
def _build_pipeline(directory: str, chunk_size: int, overlap: int, embedding_provider: str, corpus_signature: str):
    return RagPipeline.from_directory(directory, chunk_size=chunk_size, overlap=overlap)


def _corpus_signature(directory: str) -> str:
    parts = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith((".txt", ".pdf")):
            path = os.path.join(directory, fname)
            parts.append(f"{fname}:{os.path.getmtime(path)}:{os.path.getsize(path)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


question = st.text_area(
    "Your question",
    placeholder="e.g. What is dependency injection in Spring Boot?",
    height=90,
)
run_clicked = st.button("Ask", type="primary", use_container_width=True)

if run_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
    elif embedding_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI embeddings need OPENAI_API_KEY — enter one in the sidebar first.")
    elif llm_provider != "mock" and not os.getenv(f"{llm_provider.upper()}_API_KEY"):
        st.error(f"{llm_provider} generation needs {llm_provider.upper()}_API_KEY — enter one in the sidebar first.")
    else:
        try:
            signature = _corpus_signature(DOCS_DIR)
            pipeline = _build_pipeline(DOCS_DIR, chunk_size, overlap, embedding_provider, signature)
            with st.spinner("Retrieving and generating…"):
                result = pipeline.answer(question, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Query failed: {exc}")
            result = None

        if result is not None:
            st.markdown(f'<div class="rag-answer">{result.answer}</div>', unsafe_allow_html=True)

            if result.sources:
                st.divider()
                st.subheader("Retrieved chunks")
                for chunk, score in result.sources:
                    with st.expander(f"{chunk.id}"):
                        st.markdown(f'<span class="rag-score">score: {score:.3f}</span>', unsafe_allow_html=True)
                        st.write(chunk.text)
            else:
                st.caption("No chunks were retrieved — the documents/ folder may be empty.")
