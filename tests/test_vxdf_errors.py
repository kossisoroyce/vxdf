"""Unit tests for VXDF error handling using pytest.

These tests cover the most common failure modes:
• duplicate document IDs
• incorrect embedding dimension
• checksum corruption
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import pytest

from vxdf.writer import VXDFWriter
from vxdf.reader import VXDFReader
from vxdf import errors as vxerr


def _make_sample_file(tmp_path: Path, *, compression: str = "none") -> Path:
    file_path = tmp_path / "sample.vxdf"
    writer = VXDFWriter(str(file_path), embedding_dim=3, compression=compression)
    writer.add_chunk({"id": "a", "text": "foo", "vector": [0, 0, 0]})
    writer.add_chunk({"id": "b", "text": "bar", "vector": [1, 1, 1]})
    writer.close()
    return file_path


def test_duplicate_document_id(tmp_path: Path) -> None:
    file_path = tmp_path / "dup.vxdf"
    writer = VXDFWriter(str(file_path), embedding_dim=3, compression="none")
    writer.add_chunk({"id": "a", "text": "foo", "vector": [0, 0, 0]})
    with pytest.raises(vxerr.DuplicateDocumentIDError):
        writer.add_chunk({"id": "a", "text": "bar", "vector": [0, 0, 0]})
    writer.close()


def test_invalid_vector_length(tmp_path: Path) -> None:
    file_path = tmp_path / "vec_len.vxdf"
    writer = VXDFWriter(str(file_path), embedding_dim=3, compression="none")
    with pytest.raises(vxerr.InvalidChunkError):
        writer.add_chunk({"id": "x", "text": "foo", "vector": [0, 0]})  # wrong dim
    writer.close()


def test_checksum_mismatch(tmp_path: Path) -> None:
    file_path = _make_sample_file(tmp_path)

    # Corrupt first byte to invalidate checksum
    with open(file_path, "r+b") as f:
        byte = f.read(1)
        f.seek(0)
        f.write(struct.pack("B", (byte[0] + 1) % 256))

    with pytest.raises(vxerr.ChecksumMismatchError):
        VXDFReader(str(file_path))
