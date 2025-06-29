"""Utilities to merge multiple VXDF files and split a large VXDF.

Phase-1 advanced CLI helpers.

Public functions
----------------
merge(out_file, *inputs, dedupe="skip", show_progress=True)
split(in_file, *, size_bytes=None, chunks_per_file=None, show_progress=True)
"""
from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

from .reader import VXDFReader
from .writer import VXDFWriter

__all__ = ["merge", "split"]


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge(
    output_path: str | Path,
    input_paths: Sequence[str | Path],
    *,
    dedupe: str = "skip",  # skip|error|firstwins (skip duplicates seen later)
    show_progress: bool = True,
) -> None:
    """Merge *input_paths* into *output_path*.

    If *dedupe* is ``skip`` (default) duplicate IDs after the first occurrence
    are skipped. ``error`` aborts on duplicates. ``firstwins`` keeps the first
    seen (same as skip, but explicit).
    """
    if dedupe not in {"skip", "error", "firstwins"}:
        raise ValueError("dedupe must be skip|error|firstwins")

    input_paths = [Path(p) for p in input_paths]
    if not input_paths:
        raise ValueError("At least one input file must be provided")

    # Use first file to copy header info
    with VXDFReader(str(input_paths[0])) as first:
        emb_dim = first.embedding_dim or 0
        compression = first.header.get("compression", "none").lower()

    writer = VXDFWriter(str(output_path), embedding_dim=emb_dim, compression=compression)
    seen: set[str] = set()

    total_chunks = 0
    if show_progress and tqdm is not None and sys.stdout.isatty():
        total_chunks = sum(len(VXDFReader(str(fp)).offset_index) for fp in input_paths)  # quick counts
    bar_ctx = (
        tqdm(total=total_chunks, unit="chunk", desc="Merging", leave=False)
        if total_chunks and tqdm is not None and sys.stdout.isatty()
        else nullcontext()
    )

    with bar_ctx as bar:  # type: ignore
        for fp in input_paths:
            with VXDFReader(str(fp)) as reader:
                if reader.embedding_dim != emb_dim:
                    raise ValueError(
                        f"Embedding‐dim mismatch {reader.embedding_dim} vs {emb_dim} in {fp}"
                    )
                if reader.header.get("compression", "none").lower() != compression:
                    raise ValueError("Compression mismatch between source files")

                for chunk in reader.iter_chunks():
                    cid = chunk["id"]
                    if cid in seen:
                        if dedupe == "error":
                            raise RuntimeError(f"Duplicate doc id {cid} across files")
                        # skip duplicates in other modes
                        if bar is not None:
                            bar.update(1)
                        continue
                    seen.add(cid)
                    writer.add_chunk(chunk)
                    if bar is not None:
                        bar.update(1)

    writer.close()
    print(f"Merged {len(input_paths)} files → {output_path} ({len(seen)} docs).")


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _human_size(num: int) -> str:
    for unit in ["", "K", "M", "G", "T"]:
        if num < 1024:
            return f"{num:.1f}{unit}B"
        num /= 1024
    return f"{num:.1f}PB"


def split(
    input_path: str | Path,
    *,
    size_bytes: Optional[int] = None,
    chunks_per_file: Optional[int] = None,
    show_progress: bool = True,
) -> None:
    """Split *input_path* into numbered shards.

    Exactly one of *size_bytes* or *chunks_per_file* must be provided.
    Output files are written as ``<stem>-partNN.vxdf`` beside the input file.
    """
    if (size_bytes is None) == (chunks_per_file is None):
        raise ValueError("Specify either size_bytes or chunks_per_file, not both")

    input_path = Path(input_path)
    prefix = input_path.with_suffix("").name

    with VXDFReader(str(input_path)) as reader:
        emb_dim = reader.embedding_dim or 0
        compression = reader.header.get("compression", "none").lower()

        count = 0
        part = 0
        current_writer = None
        bytes_written = 0

        total_chunks = len(reader.offset_index)
        bar_ctx = (
            tqdm(total=total_chunks, unit="chunk", desc="Splitting", leave=False)
            if show_progress and sys.stdout.isatty() and tqdm is not None
            else nullcontext()
        )

        def _new_writer() -> VXDFWriter:
            nonlocal part, bytes_written
            if current_writer:
                current_writer.close()
            part += 1
            bytes_written = 0
            out_file = input_path.with_name(f"{prefix}-part{part:02d}.vxdf")
            return VXDFWriter(str(out_file), embedding_dim=emb_dim, compression=compression)

        current_writer = _new_writer()

        with bar_ctx as bar:  # type: ignore
            for chunk in reader.iter_chunks():
                if size_bytes is not None and bytes_written >= size_bytes:
                    current_writer = _new_writer()
                if chunks_per_file is not None and count % chunks_per_file == 0 and count != 0:
                    current_writer = _new_writer()

                current_writer.add_chunk(chunk)
                # rough bytes tally: size of msgpack blob + overhead
                bytes_written += len(json.dumps(chunk).encode()) + 16
                count += 1
                if bar is not None:
                    bar.update(1)

        current_writer.close()
    print(
        f"Split {input_path} into {part} files (size~{_human_size(size_bytes or 0)} / chunks {chunks_per_file or '-'})"
    )
