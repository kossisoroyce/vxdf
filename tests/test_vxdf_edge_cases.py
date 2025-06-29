"""Additional edge-case tests for VXDF error handling.

These tests deliberately corrupt VXDF files or exercise unusual paths to
ensure every custom error class is raised under the right conditions.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from vxdf.writer import VXDFWriter
from vxdf.reader import VXDFReader
from vxdf import errors as vxerr

FOOTER_MARKER = b"---VXDF_END---\n"
HEADER_MARKER = b"\n---HEADER_END---\n"


def _make_sample_file(tmp_path: Path, *, compression: str = "none") -> Path:
    """Create a very small VXDF file for mutation tests."""
    file_path = tmp_path / "edge.vxdf"
    writer = VXDFWriter(str(file_path), embedding_dim=3, compression=compression)
    writer.add_chunk({"id": "a", "text": "foo", "vector": [0, 0, 0]})
    writer.add_chunk({"id": "b", "text": "bar", "vector": [1, 1, 1]})
    writer.close()
    return file_path


def test_missing_footer_marker(tmp_path: Path) -> None:
    """If the VXDF end marker is missing, reader must raise InvalidFooterError."""
    file_path = _make_sample_file(tmp_path)

    # Truncate the footer marker completely
    with open(file_path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.truncate(size - len(FOOTER_MARKER))

    with pytest.raises((vxerr.InvalidFooterError, vxerr.ChecksumMismatchError)):
        VXDFReader(str(file_path))


def test_malformed_footer_json(tmp_path: Path) -> None:
    """Corrupt a byte inside the footer JSON so JSON decoding fails."""
    file_path = _make_sample_file(tmp_path)

    # Overwrite the first byte of the footer JSON ('{' -> '|')
    with open(file_path, "r+b") as f:
        # Read up to the last 512 bytes (or entire file if smaller) to locate last '{'
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        read_size = min(512, file_size)
        f.seek(-read_size, os.SEEK_END)
        tail = f.read(read_size)
        brace_pos = tail.rfind(b"{")
        assert brace_pos != -1, "Expected to find '{' in footer JSON"
        file_end_offset = f.tell()  # end after reading
        corrupt_pos = file_end_offset - (512 - brace_pos)
        f.seek(corrupt_pos)
        f.write(b"|")  # Replace '{' with an invalid char

    with pytest.raises((vxerr.InvalidFooterError, vxerr.ChecksumMismatchError)):
        VXDFReader(str(file_path))


def test_chunk_not_found(tmp_path: Path) -> None:
    """Requesting a non-existent document ID should raise ChunkNotFoundError."""
    file_path = _make_sample_file(tmp_path)
    reader = VXDFReader(str(file_path))
    with pytest.raises(vxerr.ChunkNotFoundError):
        reader.get_chunk("nonexistent-id")
    reader.file.close()
