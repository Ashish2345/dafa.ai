"""
Domain models representing business entities.

These are the core business objects used throughout the application.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models.enums import FileType, ParseStatus


@dataclass
class DocumentMetadata:
    """Metadata about an uploaded document."""

    filename: str
    file_type: FileType
    file_size: int
    mime_type: str
    checksum: Optional[str] = None
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ParsedDocument:
    """Represents a parsed document with extracted content."""

    document_id: str
    file_path: Path
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count: Optional[int] = None
    tables: Optional[List[Dict]] = None
    images: Optional[List[bytes]] = None
    raw_data: Optional[Any] = None
    parsed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert ParsedDocument to dictionary."""
        return {
            "document_id": self.document_id,
            "file_path": str(self.file_path),
            "content": self.content,
            "metadata": self.metadata,
            "page_count": self.page_count,
            "tables": self.tables,
            "parsed_at": self.parsed_at.isoformat(),
        }


@dataclass
class DocumentRecord:
    """Complete document record including parse status and results."""

    document_id: str
    filename: str
    file_type: FileType
    file_size: int
    mime_type: str
    storage_path: str
    uploaded_at: datetime
    status: ParseStatus
    parse_result: Optional[Dict[str, Any]] = None
    parsed_at: Optional[datetime] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentRecord to dictionary for storage."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type.value,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "storage_path": self.storage_path,
            "uploaded_at": self.uploaded_at,
            "status": self.status.value,
            "parse_result": self.parse_result,
            "parsed_at": self.parsed_at,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
