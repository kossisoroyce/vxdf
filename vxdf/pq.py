"""vxdf.pq
A minimal Product Quantisation (PQ) implementation tailored for VXDF.

Goal: provide a dependency-light ANN accelerator that can be embedded directly
in a VXDF footer.  This first version keeps things intentionally simple and
implements a *flat* PQ scan (no IVF/HNSW).  The focus is on correctness and
portability – later we can drop in a SIMD-accelerated Cython extension.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _learn_kmeans(x: np.ndarray, k: int, n_iter: int = 20, seed: int = 42) -> np.ndarray:  # noqa: D401
    """Very small planar k-means (Lloyd) for educational use.

    * x: (n, d) float32
    * k: number of centroids (256 for PQ)

    Returns (k, d) float32 centroids.
    """
    n, d = x.shape
    rng = np.random.default_rng(seed)
    if n >= k:
        centroids = x[rng.choice(n, size=k, replace=False)].copy()
    else:
        # Too few samples – duplicate points to reach k centroids deterministically
        reps = np.tile(x, (int(np.ceil(k / n)), 1))[:k]
        centroids = reps.copy()

    for _ in range(n_iter):
        # Compute squared distances to centroids
        # (n, k) = (n, 1, d) - (k, d) -> broadcast
        diff = x[:, None, :] - centroids[None, :, :]
        dist2 = np.sum((diff.astype(np.float64)) ** 2, axis=-1)
        nearest = dist2.argmin(axis=1)
        # Update centroids
        for idx in range(k):
            mask = nearest == idx
            if mask.any():
                centroids[idx] = x[mask].mean(axis=0)
    return centroids.astype(np.float32)


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

class PQIndexer:
    """Learns a Product Quantiser and encodes vectors to compact uint8 codes."""

    def __init__(self, m: int = 16) -> None:
        if m <= 0:
            raise ValueError("m must be a positive integer")
        self.m = m
        self.ksub = 256  # 8-bit codes
        # dsub will be inferred from training data in `fit()`
        self.dsub: int | None = None
        self.codebooks: np.ndarray | None = None  # (m, 256, dsub)

    # ---------------------------------------------------------------------
    # Training / encoding
    # ---------------------------------------------------------------------

    def fit(self, vecs: np.ndarray, *, n_iter: int = 25) -> None:
        """Learn codebooks from training vectors (shape (N, D)).

        *D* must be divisible by *m*. We compute *dsub = D // m* automatically.
        """
        d = int(vecs.shape[1])
        if d % self.m != 0:
            raise ValueError(f"Embedding dimension {d} not divisible by m={self.m}")
        self.dsub = d // self.m

        codebooks = np.empty((self.m, self.ksub, self.dsub), dtype=np.float32)
        for i in range(self.m):
            subvecs = vecs[:, i * self.dsub : (i + 1) * self.dsub]
            codebooks[i] = _learn_kmeans(subvecs, self.ksub, n_iter=n_iter)
        self.codebooks = codebooks

    def encode(self, vecs: np.ndarray) -> np.ndarray:
        """Encode vectors to uint8 codes using learnt codebooks."""
        if self.codebooks is None or self.dsub is None:
            raise RuntimeError("fit() must be called before encode()")
        codes = np.empty((vecs.shape[0], self.m), dtype=np.uint8)
        for i in range(self.m):
            subvecs = vecs[:, i * self.dsub : (i + 1) * self.dsub]            # Compute L2 distance to 256 centroids
            diff = subvecs[:, None, :] - self.codebooks[i][None, :, :]
            dist2 = np.sum((diff.astype(np.float64)) ** 2, axis=-1)
            codes[:, i] = dist2.argmin(axis=1).astype(np.uint8)
        return codes

    # ------------------------------------------------------------------
    # IO helpers – binary layout: JSON header + raw numpy dumps
    # ------------------------------------------------------------------

    def dump(self, fp: io.BufferedWriter) -> None:
        """Write codebooks to an already-open binary file handle.

        The caller is responsible for writing codes separately (they can be
        streamed), but we put a small JSON header so the reader knows sizes.
        Layout:
            header_len (4 bytes, big-endian uint32)
            header_json (UTF-8)
            codebooks (m * 256 * dsub * 4 bytes float32)
        """
        if self.codebooks is None:
            raise RuntimeError("No codebooks – call fit() first")

        header: Dict[str, Any] = {
            "m": self.m,
            "dsub": self.dsub,
            "ksub": self.ksub,
        }
        header_bytes = json.dumps(header).encode("utf-8")
        fp.write(struct.pack(">I", len(header_bytes)))
        fp.write(header_bytes)
        fp.write(self.codebooks.tobytes(order="C"))

    @classmethod
    def load(cls, fp: io.BufferedReader) -> "PQIndexer":  # noqa: D401
        """Read codebooks from a binary file handle and return instance."""
        header_len_bytes = fp.read(4)
        (header_len,) = struct.unpack(">I", header_len_bytes)
        header = json.loads(fp.read(header_len).decode("utf-8"))
        m = header["m"]
        dsub = header["dsub"]
        ksub = header["ksub"]
        inst = cls(m=m)
        inst.dsub = dsub
        inst.ksub = ksub
        total = m * ksub * dsub
        inst.codebooks = np.frombuffer(fp.read(total * 4), dtype=np.float32).reshape(m, ksub, dsub)
        return inst


class PQSearcher:
    """Flat PQ scanner using pre-computed LUTs.

    This pure-NumPy fallback is ~1 µs per vector per block (good enough for
    ≤100 k vectors).  We can accelerate later with a Cython or Numpy-vectorised
    LUT gather.
    """

    def __init__(self, codebooks: np.ndarray, codes: np.ndarray):
        self.codebooks = codebooks  # (m, 256, dsub)
        self.codes = codes  # (N, m) uint8
        self.m, self.ksub, self.dsub = codebooks.shape
        self.vecs_per_block = codes.shape[0]

    # --------------------------------------------------------------
    def _compute_lut(self, query: np.ndarray) -> np.ndarray:  # (m, 256)
        luts = np.empty((self.m, self.ksub), dtype=np.float32)
        for i in range(self.m):
            diff = self.codebooks[i] - query[i * self.dsub : (i + 1) * self.dsub]
            luts[i] = -np.sum((diff.astype(np.float64)) ** 2, axis=1)  # negative L2 as similarity
        return luts

    def search(self, query: np.ndarray, k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) of top-k nearest codes. query shape (128,)"""
        if query.shape[0] != self.m * self.dsub:
            raise ValueError("Query dim mismatch")
        luts = self._compute_lut(query.astype(np.float32))
        # Accumulate similarity scores
        sims = np.zeros(self.codes.shape[0], dtype=np.float32)
        for block in range(self.m):
            sims += luts[block][self.codes[:, block]]
        idx = np.argpartition(-sims, kth=k)[:k]
        order = np.argsort(-sims[idx])
        return idx[order], sims[idx][order]


# ---------------------------------------------------------------------------
# Convenience API for VXDF reader
# ---------------------------------------------------------------------------

def build_pq_index(vecs: np.ndarray, m: int = 16) -> Tuple[PQIndexer, np.ndarray]:
    """Learn PQ codebooks and encode *vecs*.

    If the requested *m* does not divide the embedding dimension, we fall back
    to the largest divisor ≤16 (or 1 as a last resort). This keeps tests with
    toy dimensions (e.g. 3) working while still exercising the pipeline.
    """
    d = int(vecs.shape[1])
    if d % m != 0:
        for cand in (16, 8, 4, 2, 1):
            if d % cand == 0:
                m = cand
                break
    pq = PQIndexer(m)
    pq.fit(vecs)
    codes = pq.encode(vecs)
    return pq, codes


def load_pq_from_file(path: str | Path) -> Tuple[PQIndexer, np.ndarray]:
    """Load codebooks + codes from separate files (future use)."""
    path = Path(path)
    with path.open("rb") as f:
        pq = PQIndexer.load(f)
        codes = np.fromfile(f, dtype=np.uint8).reshape(-1, pq.m)
    return pq, codes
