"""
Pydantic schemas for API request and response models.

These models define the contract between the API and clients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import ParseStatus, ParsingType


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Current timestamp")
    version: str = Field(default="0.1.0", description="API version")


class HealthReadyResponse(BaseModel):
    """Readiness check response with dependency status."""

    status: str = Field(..., description="Service status")
    database: bool = Field(..., description="Database connection status")
    storage: bool = Field(..., description="Storage backend status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Current timestamp")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    status: str = Field(..., description="Result status (e.g. 'fail')")
    status_code: int = Field(..., description="HTTP status code")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "REQ_VALIDATION_ERROR",
                "message": "File size exceeds maximum allowed",
                "status": "fail",
                "status_code": 400,
            }
        }
    )


# --- ParsedResponse (from schema.json) ---

FileTypeLiteral = Literal["pdf", "img"]

EmbedTypeLiteral = Literal["figure", "chart", "signature", "stamp", "checkbox"]


class Embed(BaseModel):
    """Embedded content within a block (figure, chart, signature, etc.)."""

    embed_type: EmbedTypeLiteral = Field(..., description="Type of embedded content")
    embed_id: str = Field(..., description="Unique identifier for the embed")
    embed_parsed_content: Optional[str] = Field(None, description="Parsed content (text, html, markdown)")
    embed_raw_content: Optional[str] = Field(None, description="Raw content (URL or base64)")
    embed_bbox: Optional[List[float]] = Field(None, description="Bounding box of the embedded data")
    embed_cells: Optional[Any] = Field(None, description="Excel cell start, end and spans")
    embed_html_tags: Optional[Any] = Field(None, description="HTML tags where the embed exists")


class Block(BaseModel):
    """Content block within a page."""

    block_id: str = Field(..., description="Unique block identifier")
    block_parsed_content: Optional[str] = Field(None, description="Parsed text of the block")
    embeds: List[Embed] = Field(default_factory=list, description="Embeds within the block")
    block_bbox: Optional[List[float]] = Field(None, description="Bounding box of the block")
    block_cells: Optional[Any] = Field(None, description="Excel cell start, end and spans")
    block_html_tags: Optional[Any] = Field(None, description="HTML tags for the block")
    block_raw_content: Optional[str] = Field(None, description="Raw content of the block")


class Page(BaseModel):
    """Single page/sheet with content and blocks."""

    page_number: Optional[int] = Field(None, description="Page number")
    page_name: Optional[str] = Field(None, description="Page name (e.g. sheet name for Excel)")
    page_content: Optional[str] = Field(None, description="Parsed text for the page")
    page_raw_content: Optional[str] = Field(None, description="Raw content of the page")
    embeds: List[Embed] = Field(default_factory=list, description="Embeds on the page")
    content_bbox: Optional[List[float]] = Field(None, description="Bounding box of the whole content")
    content_cells: Optional[Any] = Field(None, description="Excel cell start, end and spans")
    content_html_tags: Optional[Any] = Field(None, description="HTML tags for page content")
    blocks: List[Block] = Field(default_factory=list, description="Content blocks")


class ParsedResponse(BaseModel):
    """Structured parsing result matching schema.json."""

    content: str = Field(..., description="Whole parsed text")
    raw_info: Optional[str] = Field(None, description="Raw data structure (e.g. OCR for PDF)")
    file_type: FileTypeLiteral = Field(..., description="Document format (pdf, img)")
    file_url: Optional[str] = Field(None, description="Source file URL")
    file_metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata")
    parsing_type: ParsingType = Field(..., description="Parsing method used")
    pages: List[Page] = Field(default_factory=list, description="Pages/sheets with content and blocks")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "<whole parsed text>",
                "raw_info": "<raw data structure>",
                "file_type": "pdf",
                "file_url": "https://example.com/doc.pdf",
                "file_metadata": {},
                "parsing_type": "ocr",
                "pages": [],
            }
        }
    )


# --- ParseResponse (API response with status) ---


class ParseResponse(BaseModel):
    """Response with parsing results."""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    status: ParseStatus = Field(..., description="Parsing status")
    content: Optional[str] = Field(None, description="Extracted text content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    page_count: Optional[int] = Field(None, description="Number of pages")
    tables: Optional[List[Dict]] = Field(None, description="Extracted tables")
    parsed_at: Optional[datetime] = Field(None, description="Parse completion timestamp")
    error: Optional[str] = Field(None, description="Error message if parsing failed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "document.pdf",
                "status": "COMPLETED",
                "content": "Extracted document text...",
                "metadata": {"author": "John Doe", "title": "Sample Document"},
                "page_count": 5,
                "tables": [],
                "parsed_at": "2026-01-29T10:31:00Z",
                "error": None,
            }
        }
    )


# ---------------------------------------------------------------------------
# Custom page-index tree upload (used by POST /documents/upload when the user
# supplies their own tree instead of letting the LLM generate one).
# ---------------------------------------------------------------------------


class PageIndexNodeUpload(BaseModel):
    """One node in a user-supplied page-index tree.

    System-computed fields (``id``, ``start_char``, ``end_char``, ``page_bboxes``)
    are intentionally *not* present on this model — they are ignored/overwritten
    by the ingestion pipeline even if supplied.
    """

    nodeId: str = Field(..., min_length=1, description="Dotted id like '1.2.3'")
    title: str = Field(..., min_length=1)
    summary: str = Field(default="", description="Short summary; may be empty")
    page_range: list[int] = Field(..., description="[start_page, end_page], both 1-indexed")
    children: list["PageIndexNodeUpload"] = Field(default_factory=list)

    @field_validator("page_range")
    @classmethod
    def _check_page_range(cls, v: list[int]) -> list[int]:
        if len(v) != 2:
            raise ValueError("page_range must be a 2-element list [start, end]")
        start, end = v
        if start < 1 or end < 1:
            raise ValueError("page_range values must be >= 1 (pages are 1-indexed)")
        if start > end:
            raise ValueError(f"page_range start ({start}) cannot exceed end ({end})")
        return v


class PageIndexTreeUpload(BaseModel):
    """Top-level shape of a user-supplied page-index tree JSON file."""

    document_title: str = Field(..., min_length=1)
    language: str = Field(default="en", description="e.g. 'en' or 'ne'")
    nodes: list[PageIndexNodeUpload] = Field(..., min_length=1)


PageIndexNodeUpload.model_rebuild()

