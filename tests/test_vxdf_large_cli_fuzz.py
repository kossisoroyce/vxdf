"""Stress / integration / fuzz tests for VXDF.

1. Very-large (sparse) file test: ensures offsets > 4 GiB work.
2. CLI integration: `info`, `list`, `get` sub-commands succeed.
3. Hypothesis property-based round-trip across random chunks & compression.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

import pytest
from hypothesis import given, settings, strategies as st

from vxdf.writer import VXDFWriter
from vxdf.reader import VXDFReader

# -----------------------
# 1. Large sparse file (>4 GiB offset)
# -----------------------


@pytest.mark.skipif(sys.platform == "win32", reason="Sparse seek test unreliable on Windows CI")
def test_sparse_file_large_offset(tmp_path: Path) -> None:
    """Create a sparse VXDF whose first chunk sits beyond 4 GiB."""
    big_fp = tmp_path / "huge.vxdf"
    writer = VXDFWriter(str(big_fp), embedding_dim=3, compression="none")

    # Seek to 4 GiB (2**32) to make the first chunk offset large.
    writer.file.seek((1 << 32))
    writer.add_chunk({"id": "big", "text": "hello", "vector": [0, 0, 0]})
    writer.close()

    # Reader should still locate and read the chunk correctly
    # Monkeypatch checksum to avoid reading 4 GiB of zeros
    import types
    VXDFReader._verify_checksum = lambda self: None  # type: ignore[assignment]
    reader = VXDFReader(str(big_fp))
    data = reader.get_chunk("big")
    assert data["text"] == "hello"
    reader.file.close()

# -----------------------
# 2. CLI integration tests
# -----------------------

def _run_cli(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vxdf", *args], capture_output=True, text=True)


def test_cli_info_list_get(tmp_path: Path) -> None:
    fp = tmp_path / "cli.vxdf"
    w = VXDFWriter(str(fp), embedding_dim=3, compression="none")
    w.add_chunk({"id": "x", "text": "foo", "vector": [1, 2, 3]})
    w.close()

    proc_info = _run_cli(["info", str(fp)])
    assert proc_info.returncode == 0 and "Total documents" in proc_info.stdout

    proc_list = _run_cli(["list", str(fp)])
    assert proc_list.returncode == 0 and "x" in proc_list.stdout

    proc_get = _run_cli(["get", str(fp), "x"])
    assert proc_get.returncode == 0 and json.loads(proc_get.stdout)["text"] == "foo"

# -----------------------
# 3. Property-based round-trip fuzzing
# -----------------------


from hypothesis import HealthCheck

@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    compression=st.sampled_from(["none", "zlib", "zstd"]),
    chunks=st.lists(
        st.fixed_dictionaries(
            {
                "id": st.text(min_size=1, max_size=16, alphabet=st.characters(blacklist_categories=("Cs",))),
                "text": st.text(min_size=0, max_size=32),
                "vector": st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=3, max_size=3),
            }
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda d: d["id"],
    ),
)
def test_roundtrip_fuzzy(tmp_path: Path, compression: str, chunks: List[Dict[str, object]]) -> None:  # type: ignore[type-arg]
    fp = tmp_path / "fuzz.vxdf"
    writer = VXDFWriter(str(fp), embedding_dim=3, compression=compression)
    for chunk in chunks:
        writer.add_chunk(chunk)
    writer.close()

    r = VXDFReader(str(fp))
    for ch in chunks:
        assert r.get_chunk(ch["id"]) == ch
    r.file.close()
