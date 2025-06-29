"""Compression-related and header-marker edge-case tests for VXDF."""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from vxdf.writer import VXDFWriter
from vxdf.reader import VXDFReader
from vxdf import errors as vxerr

HEADER_MARKER = b"\n---HEADER_END---\n"


def _make_file(tmp_path: Path, *, compression: str) -> Path:
    fp = tmp_path / f"sample_{compression}.vxdf"
    writer = VXDFWriter(str(fp), embedding_dim=3, compression=compression)
    writer.add_chunk({"id": "a", "text": "foo", "vector": [0.0, 0.0, 0.0]})
    writer.add_chunk({"id": "b", "text": "bar", "vector": [1.0, 1.0, 1.0]})
    writer.close()
    return fp


@pytest.mark.parametrize("compression", ["zlib", "zstd"])
def test_roundtrip_compression(tmp_path: Path, compression: str) -> None:
    """Ensure compressed files write/read round-trip correctly."""
    fp = _make_file(tmp_path, compression=compression)
    r = VXDFReader(str(fp))
    a = r.get_chunk("a")
    assert a["text"] == "foo" and a["vector"] == [0.0, 0.0, 0.0]
    r.file.close()


def test_corrupted_compressed_chunk_zstd(tmp_path: Path) -> None:
    """Corrupt zstd-compressed chunk bytes → CompressionError on read."""
    fp = _make_file(tmp_path, compression="zstd")

    # Use reader to locate first chunk offset & length
    r = VXDFReader(str(fp))
    offset = r.offset_index["a"]
    r.file.seek(offset)
    length = struct.unpack(">I", r.file.read(4))[0]
    # Corrupt middle byte of compressed payload
    corrupt_pos = offset + 4 + length // 2
    r.file.close()

    with open(fp, "r+b") as f:
        f.seek(corrupt_pos)
        b = f.read(1)
        f.seek(corrupt_pos)
        f.write(struct.pack("B", (b[0] ^ 0xFF)))  # flip bits

    # Depending on how early corruption is detected, reader init OR chunk access may fail.
    with pytest.raises((vxerr.CompressionError, vxerr.ChecksumMismatchError)):
        VXDFReader(str(fp)).get_chunk("a")


def test_missing_header_marker(tmp_path: Path) -> None:
    """Remove header end marker → reader should fail early."""
    fp = _make_file(tmp_path, compression="none")

    with open(fp, "rb") as f:
        data = f.read()
    assert HEADER_MARKER in data  # sanity

    corrupted = data.replace(HEADER_MARKER, b"\n---BROKEN_HEADER---\n")
    with open(fp, "wb") as f:
        f.write(corrupted)

    with pytest.raises((vxerr.InvalidHeaderError, vxerr.ChecksumMismatchError)):
        VXDFReader(str(fp))
