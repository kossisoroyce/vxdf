from __future__ import annotations

import hashlib
import json
import struct
import zlib
from typing import Any, Dict, Optional

import msgpack
import numpy as np
from .pq import build_pq_index
import zstandard as zstd

from . import errors


class VXDFWriter:
    """Writes data to a VXDF file."""


    def __init__(self, file_path: str, embedding_dim: int, *, compression: str = "none", fields: Optional[Dict[str, str]] = None, pq_m: int = 16) -> None:
        """
        Initializes the VXDF writer.

        Args:
            file_path (str): The path to the output .vxdf file.
            embedding_dim (int): The dimension of the vector embeddings.
            compression (str): The compression method used for chunks (default: "none").
            fields (dict): A dictionary defining the data fields and their types.
        """
        if fields is None:
            fields = {
                "id": "str",
                "text": "str",
                "meta": "dict",
                "summary": "str",
                "parent": "str",
                "provenance": "dict",
                "vector": f"float32[{embedding_dim}]"  # Using a more descriptive type
            }

        self.file_path = file_path
        self.compression = compression.lower()
        self.embedding_dim = embedding_dim
        # PQ configuration (mandatory from v0.2)
        self.pq_m = pq_m
        self._vec_buffer: list[list[float]] = []
        try:
            self.zstd_c = zstd.ZstdCompressor() if self.compression == "zstd" else None
        except Exception as exc:  # pragma: no cover
            raise errors.CompressionError(f"Failed to initialise Zstandard compressor: {exc}") from exc
        self.file = open(self.file_path, 'w+b')
        self.offset_index = {}
        self.chunk_count = 0

        # --- Write Header --- #
        header = {
            "vxdf_version": "0.1",
            "embedding_dim": embedding_dim,
            "compression": self.compression,
            "fields": fields
        }
        header_bytes = json.dumps(header, indent=2).encode('utf-8')
        self.file.write(header_bytes)
        self.file.write(b'\n---HEADER_END---\n')

    def add_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """
        Adds a single data chunk to the file.

        Args:
            chunk_data (dict): A dictionary containing the data for one chunk (e.g., id, text, vector).
        """
        # Record current position as the start of the chunk
        # Validate chunk
        if 'id' not in chunk_data:
            raise errors.InvalidChunkError("Chunk must contain an 'id' field.")
        if 'vector' not in chunk_data:
            raise errors.InvalidChunkError("Chunk must contain a 'vector' field.")
        if len(chunk_data['vector']) != self.embedding_dim:
            raise errors.InvalidChunkError(
                f"Vector length {len(chunk_data['vector'])} does not match embedding_dim={self.embedding_dim}."
            )
        # Buffer vector for PQ training
        self._vec_buffer.append(chunk_data['vector'])
        if chunk_data['id'] in self.offset_index:
            raise errors.DuplicateDocumentIDError(f"Document ID '{chunk_data['id']}' already exists in file.")

        offset = self.file.tell()
        self.offset_index[chunk_data['id']] = offset

        # Serialize chunk using msgpack
        packed_chunk = msgpack.packb(chunk_data, use_bin_type=True)
        # Apply compression if enabled
        try:
            if self.compression == "zlib":
                packed_chunk = zlib.compress(packed_chunk)
            elif self.compression == "zstd":
                packed_chunk = self.zstd_c.compress(packed_chunk)
        except Exception as exc:  # pragma: no cover
            raise errors.CompressionError(f"Failed to compress chunk: {exc}") from exc

        # Write 4-byte length prefix (unsigned int, big-endian)
        self.file.write(struct.pack('>I', len(packed_chunk)))

        # Write the packed chunk
        self.file.write(packed_chunk)
        self.chunk_count += 1

    def __enter__(self) -> VXDFWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """
        Finalizes the VXDF file by writing the offset index and footer.
        """
        # --- Write Offset Index --- #
        index_offset = self.file.tell()
        index_bytes = json.dumps(self.offset_index, indent=2).encode('utf-8')
        self.file.write(index_bytes)
        self.file.write(b'\n')

        # --- Write PQ ANN section (mandatory) --- #
        pq_offset = self.file.tell()
        vecs = np.asarray(self._vec_buffer, dtype=np.float32)
        if vecs.size == 0:
            ann_meta = {"type": "none"}
        else:
            pq, codes = build_pq_index(vecs, m=self.pq_m)
            pq.dump(self.file)
            self.file.write(codes.tobytes(order="C"))
            ann_meta = {
                "type": "pq",
                "m": self.pq_m,
                "offset": pq_offset,
                "codes_shape": [int(codes.shape[0]), int(codes.shape[1])],
            }

        # Close the file to ensure all data is written to disk before checksum.
        self.file.close()

        # --- Calculate Checksum --- #
        file_hash = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            # Hash everything up to the index offset
            file_hash.update(f.read(index_offset))
        checksum = file_hash.hexdigest()

        # --- Write Footer (re-opening in append mode) --- #
        with open(self.file_path, "ab") as f:
            footer = {
                "index_offset": index_offset,
                "checksum": checksum,
                "ann": ann_meta,
            }
            footer_bytes = json.dumps(footer, indent=2).encode('utf-8')
            f.write(footer_bytes)
            # Write footer length as a 4-byte unsigned int, then the end marker.
            f.write(struct.pack('>I', len(footer_bytes)))
            f.write(b'---VXDF_END---')

        print(f"VXDF file '{self.file_path}' created successfully with {self.chunk_count} chunks.")



