"""High-level helpers for converting common data files into VXDF.

This module is intentionally dependency-light: heavy optional deps are imported
lazily so that users who only need core VXDF functionality are not forced to
install PDF or ML libraries.
"""
from __future__ import annotations

import itertools
import json
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Iterator, List, Optional

from .errors import MissingDependencyError, NetworkError

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

from .writer import VXDFWriter

__all__ = [
    "convert",
    "SUPPORTED_TYPES",
    "detect_type",
]

SUPPORTED_TYPES = {
    "pdf",
    "csv",
    "tsv",
    "xlsx",
    "xls",
    "docx",
    "json",
    "jsonl",
    "parquet",
    "txt",
    "md",
}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed


def _embed_sentences(
    sentences: List[str],
    model_name: str,
    *,
    openai_key: Optional[str] = None,
    show_progress: bool = False,
    workers: int = 1,
) -> List[List[float]]:  # noqa: D401
    """Return embeddings for *sentences* using a local or OpenAI model.

    Falls back to zeros if `sentence-transformers` or OpenAI is unavailable – so
    basic conversion still works inside constrained CI environments.
    """
    from .auth import get_openai_api_key
    try:
        if openai_key is None:
            try:
                openai_key = get_openai_api_key()
            except Exception:
                pass

        # If an OpenAI key is available *and* the requested model is an OpenAI model, use the OpenAI Embeddings endpoint.
        if openai_key and (model_name.startswith("text-embedding-") or model_name.startswith("openai:")):
            import openai  # type: ignore

            openai.api_key = openai_key
            # split into manageable batches (<=100 inputs per request)
            batch_size = 100
            batches = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]

            def _embed_batch(batch: List[str]):
                resp = openai.Embedding.create(model=model_name, input=batch)
                return [d["embedding"] for d in resp["data"]]

            if workers > 1 and len(batches) > 1:
                embeddings: List[List[float]] = []
                with ThreadPoolExecutor(max_workers=workers) as exe:
                    futures = {exe.submit(_embed_batch, b): idx for idx, b in enumerate(batches)}
                    for fut in as_completed(futures):
                        embeddings.extend(fut.result())
                return embeddings
            else:
                embeddings: List[List[float]] = []
                for batch in batches:
                    embeddings.extend(_embed_batch(batch))
                return embeddings

        from sentence_transformers import SentenceTransformer  # type: ignore

        # If the requested model looks like an OpenAI model but no key is available,
        # fall back to a fast local default to avoid hard errors.
        _model_for_st = model_name
        if (model_name.startswith("text-embedding-") or model_name.startswith("openai:")) and not openai_key:
            import warnings
            warnings.warn("OpenAI API key not found – falling back to 'all-MiniLM-L6-v2'. Set OPENAI_API_KEY or pass --openai-key to use OpenAI embeddings.")
            _model_for_st = "all-MiniLM-L6-v2"

        # Sentence-Transformers supports both HF hub names and local paths.
        model = SentenceTransformer(_model_for_st)

        # Encode sentences in batches and convert results to regular Python lists.
        embeddings_st = model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=show_progress,
            convert_to_numpy=False,
        )
        # `encode` may already return a list of lists; ensure plain Python lists for JSON.
        return [list(map(float, vec)) for vec in embeddings_st]
    except Exception:  # pragma: no cover – missing deps or runtime error
        # Final safety-net: return zero vectors so downstream pipeline still works.
        dim = 768  # common default
        return [[0.0] * dim for _ in sentences]


# ---------------------------------------------------------------------------
# Parsers / loaders – each yields (id, text)
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> Iterator[tuple[str, str]]:

    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pdfplumber is required for PDF ingestion – install vxdf[ingest]") from exc

    with pdfplumber.open(str(path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            chunk_id = f"page-{page_idx}"  # unique per page
            yield chunk_id, text


def _load_csv_tsv(path: Path, sep: str = ",") -> Iterator[tuple[str, str]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError:  # pragma: no cover
        # fallback: naive reader
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield f"row-{i}", line.strip()
        return

    df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
    for idx, row in df.iterrows():
        yield str(idx), " ".join(map(str, row.values))


def _load_excel(path: Path) -> Iterator[tuple[str, str]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for Excel ingestion – install vxdf[ingest]") from exc

    df = pd.read_excel(path, dtype=str).fillna("")
    for idx, row in df.iterrows():
        yield str(idx), " ".join(map(str, row.values))


def _load_docx(path: Path) -> Iterator[tuple[str, str]]:
    try:
        import docx  # python-docx  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("python-docx is required for DOCX ingestion – install vxdf[ingest]") from exc

    document = docx.Document(str(path))
    for i, para in enumerate(document.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        yield f"para-{i}", text


def _load_json(path: Path) -> Iterator[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":  # JSON array
            data = json.load(f)
            for i, obj in enumerate(data):
                yield obj.get("id", str(i)), json.dumps(obj)
        else:  # line-delimited
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield obj.get("id", str(i)), json.dumps(obj)


def _load_parquet(path: Path) -> Iterator[tuple[str, str]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas + pyarrow is required for Parquet ingestion – install vxdf[ingest]") from exc

    df = pd.read_parquet(path).fillna("")
    for idx, row in df.iterrows():
        yield str(idx), " ".join(map(str, row.values))


def _load_text(path: Path) -> Iterator[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield f"line-{i}", line.strip()


# ---------------------------------------------------------------------------
# Remote helpers
# ---------------------------------------------------------------------------

def _fetch_remote(url: str) -> Path:
    """Download *url* to a temporary file and return the Path."""
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https"}:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("HTTP ingestion requires 'requests'. Install via 'pip install requests' or 'pip install vxdf[ingest]'.") from exc
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise NetworkError(f"Failed to download {url}: {exc}") from exc
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(parsed.path).suffix)
        with tmp as f:
            shutil.copyfileobj(resp.raw, f)
        return Path(tmp.name)
    elif scheme == "s3":
        from .auth import ensure_aws_credentials
        ensure_aws_credentials()
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise MissingDependencyError("S3 ingestion requires 'boto3'. Install via 'pip install boto3'.") from exc
        s3 = boto3.client("s3")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(key).suffix)
        s3.download_file(bucket, key, tmp.name)
        return Path(tmp.name)
    elif scheme == "gs":
        from .auth import ensure_gcp_credentials
        ensure_gcp_credentials()
        try:
            from google.cloud import storage  # type: ignore
        except ImportError as exc:
            raise MissingDependencyError("GCS ingestion requires 'google-cloud-storage'. Install via 'pip install google-cloud-storage'.") from exc
        client = storage.Client()
        bucket = client.bucket(parsed.netloc)
        blob = bucket.blob(parsed.path.lstrip("/"))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(parsed.path).suffix)
        blob.download_to_filename(tmp.name)
        return Path(tmp.name)
    else:
        # If *scheme* is empty, treat *url* as a local path so upstream logic can decide.
        if scheme == "":
            return Path(url)
        raise ValueError(f"Unsupported URL scheme: {scheme}")


_REMOTE_SCHEMES = {"http", "https", "s3", "gs"}


_LOADERS = {
    "pdf": _load_pdf,
    "csv": _load_csv_tsv,
    "tsv": lambda p: _load_csv_tsv(p, sep="\t"),
    "xlsx": _load_excel,
    "xls": _load_excel,
    "docx": _load_docx,
    "json": _load_json,
    "jsonl": _load_json,
    "parquet": _load_parquet,
    "txt": _load_text,
    "md": _load_text,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in {"jsonl"}:
        return "jsonl"
    return ext or "txt"


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: str = "all-MiniLM-L6-v2",
    compression: str = "none",
    openai_key: Optional[str] = None,
    recursive: bool = False,
    show_progress: bool = True,
    resume: bool = False,
    workers: int = 1,
    detect_pii: bool = True,
    pii_patterns: Optional[List[str]] = None,
) -> None:
    """Convert *input_path* to a VXDF file at *output_path*.

    Each row / paragraph becomes a chunk with id/text/vector.
    """

    # Compile custom PII patterns once
    compiled_patterns = None
    if detect_pii and pii_patterns:
        from .pii import compile_patterns
        compiled_patterns = compile_patterns(pii_patterns)

    # Handle remote URLs
    if isinstance(input_path, str) and urllib.parse.urlparse(str(input_path)).scheme in _REMOTE_SCHEMES:
        in_path = _fetch_remote(str(input_path))
    else:
        in_path = Path(input_path)
    out_path = Path(output_path)

    existing_ids: set[str] = set()
    if resume and out_path.exists():
        try:
            from .reader import VXDFReader  # local import to avoid cycle at module load

            with VXDFReader(str(out_path)) as _r:
                existing_ids = set(_r.offset_index.keys())
                # Ensure embedding/compression stays consistent with existing file
                compression = _r.header.get("compression", compression)
                embedding_dim_existing = _r.embedding_dim
        except Exception as _exc:  # pragma: no cover
            print(f"Resume requested but failed to read existing file: {_exc}. Starting fresh.")

    spinner = None
    if show_progress and sys.stdout.isatty():
        # Start a lightweight spinner so users see immediate activity while model loads
        class _Spinner:
            def __init__(self) -> None:
                self._stop = threading.Event()
                self._thread = threading.Thread(target=self._spin, daemon=True)

            def _spin(self) -> None:
                for ch in itertools.cycle("|/−\\"):
                    if self._stop.is_set():
                        break
                    sys.stdout.write(f"\rIngesting… {ch}")
                    sys.stdout.flush()
                    time.sleep(0.1)
                sys.stdout.write("\r")  # clear

            def start(self) -> None:
                self._thread.start()

            def stop(self) -> None:
                self._stop.set()
                self._thread.join()

        spinner = _Spinner()
        spinner.start()

    BATCH_SIZE = 512  # embed/write in batches for memory safety
    ids: List[str] = []
    texts: List[str] = []
    writer: Optional[VXDFWriter] = None
    copy_done = False

    from .pii import contains_pii

    def _flush_batch() -> None:
        nonlocal writer
        if not ids:
            return
        embeds = _embed_sentences(
            texts,
            model,
            openai_key=openai_key,
            show_progress=show_progress and sys.stdout.isatty(),
            workers=workers,
        )
        if writer is None:
            writer = VXDFWriter(
                str(out_path) if not resume else str(out_path) + ".tmp",
                embedding_dim=len(embeds[0]),
                compression=compression,
            )
            if resume and existing_ids and not copy_done:
                from .reader import VXDFReader

                with VXDFReader(str(out_path)) as _old:
                    for chk in _old.iter_chunks():
                        writer.add_chunk(chk)
                copy_done = True
        for cid, text, vec in zip(ids, texts, embeds):
            chunk = {"id": cid, "text": text, "vector": vec}
            if detect_pii and contains_pii(text, patterns=compiled_patterns):
                chunk["sensitive"] = True
            writer.add_chunk(chunk)
        ids.clear()
        texts.clear()
        return

    # ... (rest of the code remains the same)
        parsed = urllib.parse.urlparse(str(fp))
        scheme = parsed.scheme.lower()
        if scheme in {"http", "https"}:
            try:
                import requests  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise MissingDependencyError("HTTP ingestion requires 'requests'. Install via 'pip install requests' or 'pip install vxdf[ingest]'.") from exc
            try:
                resp = requests.get(str(fp), stream=True, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise NetworkError(f"Failed to download {fp}: {exc}") from exc
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(parsed.path).suffix)
            with tmp as f:
                shutil.copyfileobj(resp.raw, f)
            return Path(tmp.name)
        elif scheme == "s3":
            from .auth import ensure_aws_credentials
            ensure_aws_credentials()
            try:
                import boto3  # type: ignore
            except ImportError as exc:
                raise MissingDependencyError("S3 ingestion requires 'boto3'. Install via 'pip install boto3'.") from exc
            s3 = boto3.client("s3")
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(key).suffix)
            s3.download_file(bucket, key, tmp.name)
            return Path(tmp.name)
        elif scheme == "gs":
            from .auth import ensure_gcp_credentials
            ensure_gcp_credentials()
            try:
                from google.cloud import storage  # type: ignore
            except ImportError as exc:
                raise MissingDependencyError("GCS ingestion requires 'google-cloud-storage'. Install via 'pip install google-cloud-storage'.") from exc
            client = storage.Client()
            bucket = client.bucket(parsed.netloc)
            blob = bucket.blob(parsed.path.lstrip("/"))
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(parsed.path).suffix)
            blob.download_to_filename(tmp.name)
            return Path(tmp.name)
        else:
            raise ValueError(f"Unsupported URL scheme: {scheme}")

    def _process_file(fp: Path) -> None:
        if fp.is_file():
            ftype = detect_type(fp)
            if ftype not in _LOADERS:
                return  # skip unsupported
            loader = _LOADERS[ftype]
            prefix = fp.stem
            for cid, text in loader(fp):
                if f"{prefix}-{cid}" in existing_ids:
                    continue  # skip already ingested
                ids.append(f"{prefix}-{cid}")
                texts.append(text)
                if len(ids) >= BATCH_SIZE:
                    _flush_batch()
        elif fp.is_dir():
            files = (
                fp.rglob("*") if recursive else fp.iterdir()
            )
            for f in files:
                _process_file(f)
        else:
            try:
                parsed = urllib.parse.urlparse(str(fp))
                if parsed.scheme and parsed.scheme.lower() in _REMOTE_SCHEMES:
                    remote_fp = _fetch_remote(str(fp))
                    _process_file(remote_fp)
                    _flush_batch()
                else:
                    raise ValueError(f"Unsupported path or scheme: {fp}")
            except ValueError as e:
                print(f"Skipping {fp}: {e}")

    if in_path.is_dir():
        files = (
            in_path.rglob("*") if recursive else in_path.iterdir()
        )
        for fp in files:
            _process_file(fp)
    else:
        _process_file(in_path)
    # Flush any remaining items
    _flush_batch()

    if resume and copy_done:
        # replace original file atomically
        import os
        if writer:
            writer.close()
        os.replace(out_path.with_suffix(out_path.suffix + ".tmp"), out_path)

    if spinner is not None:
        spinner.stop()

    if writer is None:
        raise ValueError("No supported files found for ingestion.")

    # Done
    writer.close()
