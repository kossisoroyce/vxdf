"""Streamlit RAG demo: compare company policies against EU regulations.

Run with:
    streamlit run use_case/app.py

Dependencies: streamlit, pdfplumber, sentence-transformers, openai (optional).
The app expects an EU policy VXDF file at ``use_case/eu_policies.vxdf``.
If an OpenAI key is set (env `OPENAI_API_KEY`) embeddings will default to
``text-embedding-3-large``; else falls back to ``all-MiniLM-L6-v2`` (local).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from a .env file (if present)
load_dotenv()

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from vxdf import VXDFReader
from vxdf.auth import get_openai_api_key

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

try:
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore

# ---------------------------------------------------------------------------
EU_VXDF_PATH = Path(__file__).with_suffix("").parent / "eu_policies.vxdf"

st.set_page_config(page_title="VXDF Compliance Checker", layout="wide")
st.title("📜🔍 EU Policy Compliance Checker")

# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading EU policy database…")
def _load_vxdf(path: Path) -> tuple[List[str], NDArray[np.float32]]:
    reader = VXDFReader(str(path))
    ids = list(reader.offset_index.keys())
    vecs = np.asarray([reader.get_chunk(cid)["vector"] for cid in ids], dtype=np.float32)
    return ids, vecs


def _embed(sentences: List[str]) -> NDArray[np.float32]:
    """Embed sentences using OpenAI or local SentenceTransformer."""
    api_key = get_openai_api_key()
    if api_key and openai is not None:
        openai.api_key = api_key
        resp = openai.Embedding.create(model="text-embedding-3-large", input=sentences)  # type: ignore
        return np.asarray([d["embedding"] for d in resp["data"]], dtype=np.float32)
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Install via `pip install sentence-transformers`. ")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(sentences, normalize_embeddings=True)


def _similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    # cosine sim since vectors are normalized
    return np.dot(a, b.T)

# ---------------------------------------------------------------------------
if not EU_VXDF_PATH.exists():
    st.error(
        f"EU policy VXDF not found at {EU_VXDF_PATH}. Please place the file there and restart the app.")
    st.stop()

EU_IDS, EU_VECS = _load_vxdf(EU_VXDF_PATH)

# Sidebar: upload company policy PDF or paste text -------------------------------------------------
with st.sidebar:
    st.header("Company Policy Input")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    pasted_text = st.text_area("…or paste policy text here", height=200)

    def _pdf_to_paragraphs(data: bytes) -> List[str]:
        if pdfplumber is None:
            st.warning("pdfplumber not installed; can't parse PDF.")
            return []
        paras: List[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                for para in txt.split("\n\n"):
                    para = para.strip()
                    if para:
                        paras.append(para)
        return paras

    company_paras: List[str] = []
    if uploaded is not None:
        company_paras.extend(_pdf_to_paragraphs(uploaded.read()))
    if pasted_text.strip():
        company_paras.append(pasted_text.strip())

    if company_paras:
        comp_vecs = _embed(company_paras)
    else:
        comp_vecs = np.zeros((0, EU_VECS.shape[1]), dtype=np.float32)

# Main chat interface -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about compliance…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- RAG: embed query, search EU and company docs -----------------------
    q_vec = _embed([prompt])[0]
    sims_eu = _similarity(q_vec.reshape(1, -1), EU_VECS)[0]
    top_idx = np.argsort(sims_eu)[-3:][::-1]
    eu_hits = [(EU_IDS[i], sims_eu[i]) for i in top_idx]

    context_chunks: List[str] = []
    reader = VXDFReader(str(EU_VXDF_PATH))
    for cid, score in eu_hits:
        chunk = reader.get_chunk(cid)
        context_chunks.append(f"EU:{cid} (score {score:.2f}): {chunk['text']}")

    if comp_vecs.shape[0]:
        sims_comp = _similarity(q_vec.reshape(1, -1), comp_vecs)[0]
        best_idx = int(np.argmax(sims_comp))
        best_score = float(sims_comp[best_idx])
        context_chunks.append(f"COMPANY (score {best_score:.2f}): {company_paras[best_idx][:300]}")

    context = "\n---\n".join(context_chunks)

    # Generate answer --------------------------------------------------------
    answer: str
    api_key = get_openai_api_key()
    if api_key and openai is not None:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(  # type: ignore
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a compliance assistant referencing EU regulations."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{prompt}"},
            ],
            temperature=0.2,
        )
        answer = resp["choices"][0]["message"]["content"].strip()
    else:
        # Fallback: simple extractive answer = top match text
        answer = "\n\n".join(context_chunks[:2])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
