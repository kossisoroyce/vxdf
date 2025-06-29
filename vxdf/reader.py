from __future__ import annotations

import hashlib
import json
import struct
import zlib
from typing import Any, Dict, Iterator, Optional

import msgpack
import zstandard as zstd

from . import errors


class VXDFReader:
    """Reads and parses a VXDF file."""

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

    def _parse_structure(self) -> None:
        """Parses the file to locate and load the header, index, and footer."""
        self.file.seek(0, 2)
        file_size = self.file.tell()

        search_buf_size = min(file_size, 4096)
        self.file.seek(file_size - search_buf_size)
        buffer = self.file.read(search_buf_size)

        end_marker = b'---VXDF_END---'
        end_marker_pos = buffer.rfind(end_marker)
        if end_marker_pos == -1:
            raise errors.InvalidFooterError("VXDF end marker '---VXDF_END---' not found. The file may be truncated or corrupted.")
        
        content_before_marker = buffer[:end_marker_pos]
        
        # Locate the *last* opening brace before the end marker – this starts the footer JSON.
        brace_pos = content_before_marker.rfind(b'{')
        if brace_pos == -1:
            raise errors.InvalidFooterError("Could not locate footer JSON start before end marker.")

        footer_json_str = content_before_marker[brace_pos:]
        try:
            self.footer = json.loads(footer_json_str.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise errors.InvalidFooterError(f"Failed to parse footer JSON: {exc}") from exc

        self.index_offset = self.footer['index_offset']
        self._footer_start_pos = file_size - search_buf_size + brace_pos

        # Verify checksum BEFORE attempting to decode index or header, so corruption is caught early
        self._verify_checksum()

        # --- Load Offset Index --- #
        self.file.seek(self.index_offset)
        index_bytes = self.file.read(self._footer_start_pos - self.index_offset)
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
