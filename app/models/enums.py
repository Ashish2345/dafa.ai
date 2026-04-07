"""
Enumerations used throughout the application.
"""

from enum import Enum


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class FileType(str, Enum):
    """Supported document file types."""

    PDF = "PDF"
    IMAGE = "IMAGE"
    UNKNOWN = "UNKNOWN"


class ParseStatus(str, Enum):
    """Document parsing status."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StorageBackend(str, Enum):
    """Storage backend types."""

    LOCAL = "local"
    S3 = "s3"


class ParsingType(str, Enum):
    """Parsing method used for document extraction."""

    OCR = "ocr"
    TEXT = "text"
    STRUCTURED = "structured"
    HYBRID = "hybrid"


class OCRProvider(str, Enum):
    """OCR provider options for non-digital document processing."""

    GOOGLE = "google"
    AWS = "aws"
    AZURE = "azure"
