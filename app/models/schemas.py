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

    Optional ``start_text`` / ``end_text`` fields let the user specify text
    anchors (e.g. the first and last few words of the section).  At ingestion
    the system searches for these in the OCR markdown to compute aligned
    ``start_char`` / ``end_char``, giving precise section boundaries for
    highlighting and text extraction.

    Extra user-defined fields (e.g. ``keywords``, ``tags``) are allowed and
    preserved in the saved tree so downstream tooling can use them.
    """

    model_config = ConfigDict(extra="allow")

    nodeId: str = Field(..., min_length=1, description="Dotted id like '1.2.3'")
    title: str = Field(..., min_length=1)
    summary: str = Field(default="", description="Short summary; may be empty")
    page_range: list[int] = Field(..., description="[start_page, end_page], both 1-indexed")
    children: list["PageIndexNodeUpload"] = Field(default_factory=list)
    start_text: str = Field(default="", description="First few words of the section text — used to locate the section start in OCR markdown")
    end_text: str = Field(default="", description="Last few words of the section text — used to locate the section end in OCR markdown")

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

    model_config = ConfigDict(extra="allow")

    document_title: str = Field(..., min_length=1)
    language: str = Field(default="en", description="e.g. 'en' or 'ne'")
    nodes: list[PageIndexNodeUpload] = Field(..., min_length=1)


PageIndexNodeUpload.model_rebuild()


# ---------------------------------------------------------------------------
# Pre-parsed content upload — the dump produced by POST /documents/parse.
# When the client supplies this file at upload time, the pipeline skips
# OCR + markdown conversion and persists the supplied layers as-is.
# ---------------------------------------------------------------------------


class ParsedPageWord(BaseModel):
    """One word with normalized 0-1 coordinates, matching page_ocr_bboxes."""

    model_config = ConfigDict(extra="allow")

    text: str
    x0: float
    y0: float
    x2: float
    y2: float
    char_offset: int = Field(default=0, ge=0)


class ParsedPage(BaseModel):
    """One page worth of OCR words — mirrors one row in page_ocr_bboxes."""

    page: int = Field(..., ge=1)
    words: List[ParsedPageWord] = Field(default_factory=list)


class ParsedContentLayer(BaseModel):
    markdown: str = Field(..., min_length=1)
    language: str = Field(default="en")


class ParsedRawLayer(BaseModel):
    pages: List[ParsedPage] = Field(default_factory=list)
    page_count: Optional[int] = None


class ParsedContentUpload(BaseModel):
    """Full parsed-content payload produced by ``POST /documents/parse``."""

    model_config = ConfigDict(extra="allow")

    parsed: ParsedContentLayer
    raw: ParsedRawLayer = Field(default_factory=ParsedRawLayer)


# ---------------------------------------------------------------------------
# Query endpoint models — used by both POST /query and POST /query/stream.
# ---------------------------------------------------------------------------


class ConversationMessage(BaseModel):
    """One prior turn sent by the client for follow-up context."""

    role: str
    content: str


class QueryRequest(BaseModel):
    """Non-streaming query request."""

    query: str = Field(..., description="User query/question", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = Field(default=True)
    strategy: Optional[str] = Field(
        default=None, description="'page_index' or 'vector'. Defaults to server setting."
    )
    collection_name: Optional[str] = Field(default=None)
    filter_conditions: Optional[Dict[str, Any]] = Field(default=None)


class StreamQueryRequest(QueryRequest):
    """Streaming query request — adds conversation context + language override."""

    response_language: Optional[str] = Field(
        default=None,
        description="Response language: 'en' or 'ne'. When omitted, auto-detected from content.",
    )
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default=None, description="Last few messages for follow-up context."
    )


class QueryResponse(BaseModel):
    """Non-streaming query response."""

    query: str
    answer: str
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

