"""Minimal LangChain integration example for VXDF.

Requires: pip install langchain-community sentence-transformers

This script demonstrates loading a VXDF file as a vector store in LangChain.
"""
from __future__ import annotations

from langchain_community.vectorstores import VXDF  # type: ignore
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Path to an existing VXDF file
VXDF_PATH = "demo.vxdf"

# Initialise embedding model (any model is fine as we just load pre-computed embeddings)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load store
store = VXDF.from_vxdf(VXDF_PATH, embedding_function=embeddings)

# Simple similarity search
docs = store.similarity_search("hello world", k=3)
for doc in docs:
    print(doc.metadata["id"], doc.page_content[:60])
