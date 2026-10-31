"""Streamlit sidebar: upload material, build the index, choose retrieval settings."""
from pathlib import Path

import streamlit as st

from .pipeline import build_retriever

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "sample_notes.txt"


def render_sidebar():
    """Returns (retriever_or_None, k)."""
    st.sidebar.header("1 · Your material")
    files = st.sidebar.file_uploader("Upload syllabus / notes", type=["pdf", "txt", "md"],
                                     accept_multiple_files=True)
    use_sample = st.sidebar.checkbox("Use sample notes (Intro to ML)")

    if st.sidebar.button("Build index", type="primary"):
        payload = [(f.name, f.getvalue()) for f in files or []]
        if use_sample:
            payload.append((SAMPLE.name, SAMPLE.read_bytes()))
        if not payload:
            st.sidebar.warning("Upload a file or tick the sample notes first.")
        else:
            try:
                with st.spinner("Reading, chunking and embedding..."):
                    st.session_state["retriever"] = build_retriever(payload)
                st.session_state.pop("pack", None)
            except ValueError as e:
                st.sidebar.error(str(e))

    retriever = st.session_state.get("retriever")
    if retriever:
        st.sidebar.success(f"Indexed {retriever.n_chunks} chunks from {len(retriever.sources)} file(s) "
                           f"({retriever.embedder.name} embeddings)")

    st.sidebar.header("2 · Retrieval")
    k = st.sidebar.slider("Passages retrieved (k)", 3, 12, 6)
    return retriever, k
