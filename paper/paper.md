# VXDF: A Vector-Native, Portable File Format for Lightning-Fast Retrieval-Augmented Generation

*Authors: **Your Names Here***  
*Affiliation: PolicyGuard.AI, San Francisco, CA, USA*  
*Correspondence: first.last@policyguard.ai*

## Abstract
Retrieval-Augmented Generation (RAG) systems conventionally rely on external vector databases that introduce infrastructure overhead, network latency, and vendor lock-in. **VXDF (Vector eXchange Data Format)** is a self-contained binary container that stores raw text, metadata, and high-dimensional embeddings in one portable file. VXDF delivers **2–7× faster retrieval** than popular vector stores on commodity hardware while requiring no external services. We present VXDF’s file layout, algorithms, and extensive benchmarks, demonstrating its advantages over conventional RAG solutions.

## 1 Introduction
Large Language Models (LLMs) increasingly integrate RAG pipelines to ground generation in external knowledge. Traditional stacks couple an embedding model with a vector database (e.g., FAISS, Milvus, Pinecone). These systems introduce operational and latency burdens. VXDF collapses the stack into a single file that can be version-controlled, shipped, or mounted locally, enabling immediate, offline RAG.

*…Full paper mirrors the LaTeX version (see `paper/main.tex`).*

## 2 How to Reproduce Benchmarks

```bash
# 1. Create a virtual environment and install dev deps
pip install -r requirements.txt  # root project reqs
pip install faiss-cpu seaborn rich tqdm

# 2. Run benchmarks
python benchmarks/run_benchmarks.py --output benchmarks/results.csv

# 3. Plot latency & throughput
python benchmarks/plot_latency.py --csv benchmarks/results.csv --outfig figures/latency_box.pdf
```

All benchmark scripts produce machine-readable CSV files suitable for inclusion in this paper. See `benchmarks/README.md` for details.
