from __future__ import annotations

import hashlib
import json
import struct
import zlib
from typing import Any, Dict, Iterator, Optional

import msgpack
import numpy as np
from .pq import PQIndexer, PQSearcher, build_pq_index
import zstandard as zstd

from . import errors


class VXDFReader:
    """Reads and parses a VXDF file.

    Typical chunk schema (as written by :class:`vxdf.writer.VXDFWriter` ≥ 0.2):

    * ``id`` – unique chunk/document identifier (str)
    * ``text`` – raw text content (str)
    * ``vector`` – list[float] embedding of dimension ``header["embedding_dim"]``
    * ``summary`` – *optional* one-sentence summary generated during ingestion
    * ``parent`` – *optional* parent chunk/doc id (for hierarchical corpora)
    * ``provenance`` – *optional* dictionary or string describing the source
      (e.g. original filename, URL, author, version)

    These additional metadata fields are **always preserved** by the reader –
    they are returned verbatim by :py:meth:`get_chunk` and
    :py:meth:`iter_chunks`, allowing downstream applications to display or
    reason over them without extra parsing.

    Example
    -------
    >>> from vxdf.reader import VXDFReader
    >>> with VXDFReader("sample.vxdf") as r:
    ...     first = next(r.iter_chunks())
    ...     print(first["summary"], first.get("provenance"))
    """

    def __init__(self, file_path: str) -> None:
        """Initializes the VXDF reader."""
        self.file_path = file_path
        self.file = open(self.file_path, 'rb')
        
        self._footer_start_pos = 0
        self.header = {}
        self.footer = {}
        self.offset_index = {}
        self.data_start_offset = 0
        self.compression = "none"

        self._parse_structure()
        # ANN searcher (PQ)
        self._pq_searcher: PQSearcher | None = None
        if "ann" in self.footer and self.footer["ann"].get("type") == "pq":
            self._load_pq()

    def _parse_structure(self) -> None:
        """Parses the file to locate and load the header, index, and footer."""
        self.file.seek(0, 2)
        file_size = self.file.tell()

        search_buf_size = min(file_size, 4096)
        self.file.seek(file_size - search_buf_size)
        buffer = self.file.read(search_buf_size)

        end_marker = b'---VXDF_END---'
        end_marker_len = len(end_marker)
        end_marker_pos = buffer.rfind(end_marker)
        if end_marker_pos == -1:
            raise errors.InvalidFooterError("VXDF end marker '---VXDF_END---' not found. The file may be truncated or corrupted.")

        # Footer length is stored in the 4 bytes *before* the end marker.
        footer_len_pos = end_marker_pos - 4
        footer_len_bytes = buffer[footer_len_pos:end_marker_pos]
        footer_len = struct.unpack('>I', footer_len_bytes)[0]

        # The footer itself is before its length.
        footer_pos = footer_len_pos - footer_len
        footer_json_str = buffer[footer_pos:footer_len_pos].decode("utf-8")

        try:
            self.footer = json.loads(footer_json_str)
        except json.JSONDecodeError as exc:
            raise errors.InvalidFooterError(f"Failed to parse footer JSON: {exc}") from exc

        self.index_offset = self.footer['index_offset']
        self._footer_start_pos = file_size - search_buf_size + footer_pos

        # Verify checksum BEFORE attempting to decode index or header, so corruption is caught early
        self._verify_checksum()

        # --- Load Offset Index --- #
        self.file.seek(self.index_offset)
        index_end = self.footer.get("ann", {}).get("offset", self._footer_start_pos)
        index_bytes = self.file.read(index_end - self.index_offset)
        # Index JSON may be followed by a newline – strip common whitespace before decoding.
        index_bytes = index_bytes.rstrip(b"\n\r ")
        try:
            self.offset_index = json.loads(index_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise errors.InvalidFooterError(f"Failed to parse offset index JSON: {exc}") from exc

        # --- Load Header --- #
        self.file.seek(0)
        header_search_chunk = self.file.read(4096)
        header_end_marker = b'\n---HEADER_END---\n'
        header_end_pos = header_search_chunk.find(header_end_marker)
        if header_end_pos == -1:
            raise errors.InvalidHeaderError("VXDF header end marker '---HEADER_END---' not found.")
        header_bytes = header_search_chunk[:header_end_pos]
        try:
            self.header = json.loads(header_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise errors.InvalidHeaderError(f"Failed to parse header JSON: {exc}") from exc
        self.data_start_offset = header_end_pos + len(header_end_marker)
        self.compression = self.header.get("compression", "none").lower()

    # ---------------------------------------------------------------------
    # ANN helpers
    # ---------------------------------------------------------------------

    def _ensure_pq(self) -> None:
        """Ensure that a PQ index is available.

        If the current VXDF file predates mandatory ANN support and therefore
        lacks a pre-built PQ section, we build one *in memory* on first use so
        that ``search()`` always works.  The rebuilt index is **not** persisted
        back to disk – callers who need that can open the file with
        ``VXDFWriter.update()`` or an explicit CLI command.  This keeps the
        reader side fast and side-effect-free while still gracefully handling
        legacy files.
        """
        if self._pq_searcher is not None:
            return  # already present

        # Fallback path: load all vectors and build PQ on the fly.
        vecs: list[list[float]] = []
        for doc_id in self.offset_index.keys():
            chunk = self.get_chunk(doc_id)
            vecs.append(chunk["vector"])
        if not vecs:
            raise RuntimeError("Cannot build PQ index – file contains no vectors.")

        vecs_np = np.asarray(vecs, dtype=np.float32)
        pq, codes = build_pq_index(vecs_np)
        self._pq_searcher = PQSearcher(pq.codebooks, codes)


    def _load_pq(self) -> None:
        """Lazy-load PQ codebooks and codes from file based on footer metadata."""
        meta = self.footer["ann"]
        offset = meta["offset"]
        codes_shape = meta["codes_shape"]
        self.file.seek(offset)
        pq = PQIndexer.load(self.file)
        # After loader returns, file pointer is after codebooks. The rest are codes.
        codes_bytes = np.frombuffer(self.file.read(codes_shape[0] * codes_shape[1]), dtype=np.uint8)
        codes = codes_bytes.reshape(codes_shape)
        self._pq_searcher = PQSearcher(pq.codebooks, codes)

    # ---------------------------------------------------------------------
    def _verify_checksum(self) -> None:
        """Verifies the integrity of the file using the SHA256 checksum."""
        self.file.seek(0)
        content_to_hash = self.file.read(self.index_offset)
        calculated_hash = hashlib.sha256(content_to_hash).hexdigest()

        if calculated_hash != self.footer['checksum']:
            raise errors.ChecksumMismatchError("SHA-256 checksum mismatch: file may be corrupted.")

    def get_chunk(self, doc_id: str) -> Dict[str, Any]:
        """Retrieves a single chunk by its document ID."""
        if doc_id not in self.offset_index:
            raise errors.ChunkNotFoundError(f"Document ID '{doc_id}' not found in the index.")

        offset = self.offset_index[doc_id]
        self.file.seek(offset)

        length_bytes = self.file.read(4)
        chunk_length = struct.unpack('>I', length_bytes)[0]

        packed_chunk = self.file.read(chunk_length)
        try:
            if self.compression == "zlib":
                packed_chunk = zlib.decompress(packed_chunk)
            elif self.compression == "zstd":
                packed_chunk = zstd.decompress(packed_chunk)
        except Exception as exc:
            raise errors.CompressionError(f"Failed to decompress chunk: {exc}") from exc
        return msgpack.unpackb(packed_chunk, raw=False)

    def iter_chunks(self) -> Iterator[Dict[str, Any]]:
        """Yields all data chunks in the order they appear in the file."""
        sorted_offsets = sorted(self.offset_index.values())
        for offset in sorted_offsets:
            self.file.seek(offset)
            length_bytes = self.file.read(4)
            chunk_length = struct.unpack('>I', length_bytes)[0]
            packed_chunk = self.file.read(chunk_length)
            try:
                if self.compression == "zlib":
                    packed_chunk = zlib.decompress(packed_chunk)
                elif self.compression == "zstd":
                    packed_chunk = zstd.decompress(packed_chunk)
            except Exception as exc:
                raise errors.CompressionError(f"Failed to decompress chunk: {exc}") from exc
            yield msgpack.unpackb(packed_chunk, raw=False)

    # ---------------------------------------------------------------------
    # Public ANN search API
    # ---------------------------------------------------------------------
    def search(self, query_vec: np.ndarray, k: int = 4):
        """Return top-k (doc_id, score) pairs for a query vector."""
        # Lazy-create PQ index for older files if needed.
        if self._pq_searcher is None:
            self._ensure_pq()
        idxs, sims = self._pq_searcher.search(query_vec, k)
        # Map indices to doc_ids via offset_index order
        doc_ids = list(self.offset_index.keys())
        return [(doc_ids[i], float(sims_i)) for i, sims_i in zip(idxs, sims)]

    # ---------------------------------------------------------------------
    def close(self) -> None:
        """Closes the file handle."""
        self.file.close()

    def __enter__(self) -> VXDFReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def embedding_dim(self) -> Optional[int]:
        return self.header.get('embedding_dim')

    @property
    def vxdf_version(self) -> Optional[str]:
        return self.header.get('vxdf_version')
