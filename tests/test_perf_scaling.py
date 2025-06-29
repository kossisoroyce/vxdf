"""Simple regression test to catch accidental O(N^2) behaviour.

We time writing/reading 2k vs 4k chunks and assert the runtime scales ~linearly.
The thresholds are loose to stay green on CI, but will fail dramatically if
quadratic behaviour sneaks in.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from vxdf.reader import VXDFReader
from vxdf.writer import VXDFWriter


def _gen_chunks(n: int):
    for i in range(n):
        yield {"id": f"id{i}", "text": "x" * 10, "vector": [0.0, 0.0, 0.0]}


@pytest.mark.parametrize("n_small,n_big", [(2000, 4000)])
def test_write_read_scaling(tmp_path: Path, n_small: int, n_big: int) -> None:
    """Ensure doubling chunks does not blow up beyond ~2.5× time (linear-ish)."""

    def write_file(count: int, fp: Path):
        w = VXDFWriter(str(fp), embedding_dim=3, compression="none")
        for ch in _gen_chunks(count):
            w.add_chunk(ch)
        w.close()

    def read_file(fp: Path):
        r = VXDFReader(str(fp))
        for _ in r.iter_chunks():
            pass
        r.file.close()

    # --- small ---
    fp_small = tmp_path / "small.vxdf"
    start = time.perf_counter()
    write_file(n_small, fp_small)
    read_file(fp_small)
    t_small = time.perf_counter() - start

    # --- big (double) ---
    fp_big = tmp_path / "big.vxdf"
    start = time.perf_counter()
    write_file(n_big, fp_big)
    read_file(fp_big)
    t_big = time.perf_counter() - start

    # Allow some overhead but flag >2.5x as suspicious (quadratic would be 4x)
    assert t_big / t_small < 2.5, f"Performance scaling degraded: {t_small:.2f}s -> {t_big:.2f}s"
