# VXDF Python Library  

![PyPI](https://img.shields.io/pypi/v/vxdf?color=brightgreen) ![CI](https://github.com/kossisoroyce/vxdf/actions/workflows/ci.yml/badge.svg)

VXDF (Vector **eXchange** Data Format) is an AI-native container for text, metadata and vector embeddings—portable, indexable and compressed.  If you do RAG, semantic search or compliance audits, VXDF gives you **one file, one command**.

## Quick-start

```bash
pip install vxdf[zstd]        # installs optional Zstandard support

python - << 'PY'
from vxdf import VXDFWriter, VXDFReader

# create a small file
data = [
    {"id": "1", "text": "hello", "vector": [0.1, 0.2]},
    {"id": "2", "text": "world", "vector": [0.3, 0.4]},
]
with VXDFWriter("demo.vxdf", embedding_dim=2, compression="zstd") as w:
    for chunk in data:
        w.add_chunk(chunk)

# read it back
a = VXDFReader("demo.vxdf")
print(a.get_chunk("2"))
PY
```

## Command-line

```bash
vxdf pack data.jsonl data.vxdf --compression zstd   # create
vxdf info data.vxdf                                  # header & stats
vxdf list data.vxdf | head                           # ids
vxdf get  data.vxdf some-id > doc.json               # extract
```

## Colab / Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kossisoroyce/vxdf/blob/main/notebooks/QuickStart.ipynb)

## LangChain integration (preview)


```python
from langchain_community.vectorstores import VXDF
vs = VXDF.from_vxdf("demo.vxdf")
```

See `examples/langchain_integration.py` for a minimal adapter.

## Authentication

VXDF commands that interact with cloud services need credentials.

### OpenAI embeddings
The client looks for an API key in this order (first match wins):

1. `--openai-key` CLI flag (e.g. `vxdf convert my.pdf out.vxdf --model openai --openai-key sk-...`)
2. `OPENAI_API_KEY` environment variable.
3. `~/.vxdf/config.toml` under the `[openai]` table:

```toml
[openai]
api_key = "sk-..."
```

### AWS (S3 URLs)
Uses the standard AWS credential chain provided by *boto3* – environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), the AWS CLI config, or an attached IAM role. Run `aws configure` if unsure.

### GCP (gs:// URLs)
Relies on Application Default Credentials. Run `gcloud auth application-default login` or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing at a JSON key file.

If credentials are missing VXDF exits early with a clear message and a hint on how to configure them.

## Shell completion

Install extra dependencies and activate once:

```bash
pip install vxdf[completion]
activate-global-python-argcomplete --user  # bash/zsh/fish supported
```

Re-open your terminal and enjoy TAB-completion for `vxdf` sub-commands and options.

---
VXDF is BSD-3-licensed. Contributions welcome!
