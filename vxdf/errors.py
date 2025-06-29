"""Centralised VXDF exception hierarchy.

All public APIs raise these exceptions instead of bare `ValueError`, `KeyError`, etc.
This makes it straightforward for calling code to handle specific failure modes.
"""
from __future__ import annotations


class VXDFError(Exception):
    """Base class for all VXDF-related errors."""


class InvalidHeaderError(VXDFError):
    """Header is missing or malformed."""


class InvalidFooterError(VXDFError):
    """Footer or end marker is missing / malformed."""


class ChecksumMismatchError(VXDFError):
    """File checksum does not match footer value (file may be corrupted)."""


class CompressionError(VXDFError):
    """Error occurred during compression or decompression."""


class ChunkNotFoundError(VXDFError):
    """Requested document ID not present in the offset index."""


class DuplicateDocumentIDError(VXDFError):
    """Attempted to add a chunk with a duplicate document ID."""


class InvalidChunkError(VXDFError):
    """Chunk data failed validation (e.g., wrong embedding dimension)."""


class MissingDependencyError(VXDFError):
    """Optional library needed for this operation is not installed."""


class NetworkError(VXDFError):
    """Network request failed (e.g., download error, timeout)."""


class EncryptionError(VXDFError):
    """Chunk is encrypted but key is missing or decryption failed."""


class AuthenticationError(VXDFError):
    """Authentication credentials are missing or invalid."""
