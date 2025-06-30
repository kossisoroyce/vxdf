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
from typing import List, Any

from dotenv import load_dotenv

# Load environment variables from a .env file (if present)
load_dotenv()

# Workaround for Pillow/torchvision enum mismatch on some environments
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

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

st.set_page_config(page_title="VXDF Compliance Checker", page_icon="🛡️", layout="wide")

# ----------------- Global style -------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        html, body, [class*="st-"], .css-ffhzg2 {{
            font-family: 'Poppins', sans-serif;
        }}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
        }}
        .main-title {{
            font-weight: 600;
            font-size: 2.4rem;
            background: linear-gradient(90deg,#007cf0,#00dfd8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: #adb5bd;
            margin-top: -0.5rem;
            font-size: 0.95rem;
        }}
            /* Chat message bubbles */
        /* user */
        div[data-testid="stChatMessageUser"] .stMarkdown p {
            background: #007cf0;
            color: white;
            padding: 0.6rem 0.9rem;
            border-radius: 18px 18px 4px 18px;
            display: inline-block;
            max-width: 80%;
        }
        /* assistant */
        div[data-testid="stChatMessageAssistant"] .stMarkdown p {
            background: #f1f3f5;
            color: #212529;
            padding: 0.6rem 0.9rem;
            border-radius: 18px 18px 18px 4px;
            display: inline-block;
            max-width: 80%;
        }
        /* Reduce default margins */
        div[data-testid^="stChatMessage"] .stMarkdown p {
            margin-bottom: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🛡️  PolicyGuard Compliance Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Chat with EU regulations and your own company policies</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading EU policy database…")
def _load_vxdf(path: Path) -> tuple[List[str], NDArray[np.float32]]:
    reader = VXDFReader(str(path))
    ids = list(reader.offset_index.keys())
    vecs = np.asarray([reader.get_chunk(cid)["vector"] for cid in ids], dtype=np.float32)
    return ids, vecs


def _embed(sentences: List[str]) -> NDArray[np.float32]:
    """Embed sentences matching the reference embedding dimension (EU_DIM)."""

    """Embed sentences using OpenAI (v0 or v1 SDK) or local SentenceTransformer."""
    api_key = get_openai_api_key()
    # Prefer OpenAI embeddings when dims match EU_DIM and key available
    if api_key and openai is not None and EU_DIM in {1536, 3072}:
        try:
            from openai import OpenAI  # type: ignore
            client: Any = OpenAI(api_key=api_key)
            resp = client.embeddings.create(model="text-embedding-3-large", input=sentences)
            vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            if vecs.shape[1] == EU_DIM:
                return vecs
        except Exception:
            # fallthrough to local model
            pass
    # Local sentence-transformer fallback; choose model by target dim
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Install via `pip install sentence-transformers`. ")
    st_model_map = {384: "all-MiniLM-L6-v2", 768: "all-mpnet-base-v2"}
    model_name = st_model_map.get(EU_DIM, "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
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
EU_DIM = EU_VECS.shape[1]  # reference embedding dimension

# ----------------- API key onboarding ------------------------------------------
if "auth_stage" not in st.session_state:
    st.session_state["auth_stage"] = "await_key"

if st.session_state["auth_stage"] == "await_key":
    with st.sidebar:
        st.header("🔑 API Key Setup")
        st.write("Provide your OpenAI key for best-quality answers, or proceed with the built-in local model (no key required). Your key never leaves the browser session.")
        with st.form("key_form"):
            api_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-…")
            use_local = st.checkbox("Use built-in MiniLM model instead", value=not bool(api_input))
            submitted = st.form_submit_button("Start Chatting ")
        if submitted:
            if api_input:
                os.environ["OPENAI_API_KEY"] = api_input
                st.success("API key saved in session.")
                st.session_state["auth_stage"] = "ready"
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                elif hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.stop()
            elif use_local:
                st.session_state["auth_stage"] = "ready"
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                elif hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.stop()
            else:
                st.warning("Please enter a key or choose the local model option.")
    st.stop()

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
        try:
            from openai import OpenAI  # type: ignore

            client: Any = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a compliance assistant referencing EU regulations."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{prompt}"},
                ],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
        except (ImportError, AttributeError):
            openai.api_key = api_key  # type: ignore
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
