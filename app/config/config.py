"""
Parser configuration classes.

Contains Pydantic models for configuring document parsers.
"""

import os
from typing import List, Optional

from pydantic import BaseModel, Field

from app.config.request_params import RequestParam


class ParserConfig(BaseModel):
    """Base configuration for all parsers."""

    timeout: int = Field(default=300, description="Parse timeout in seconds")
    max_file_size: int = Field(default=52428800, description="Max file size in bytes (default 50MB)")
    extract_metadata: bool = Field(
        default=True,
        description="Extract document metadata",
        json_schema_extra={"request_param": RequestParam.EXTRACT_METADATA},
    )
    extract_tables: bool = Field(
        default=True,
        description="Extract tables from document",
        json_schema_extra={"request_param": RequestParam.EXTRACT_TABLES},
    )


class PDFParserConfig(ParserConfig):
    """Configuration for PDF parser."""

    ocr_enabled: bool = Field(
        default=True,
        description="Enable OCR for scanned PDFs",
        json_schema_extra={"request_param": RequestParam.OCR_ENABLED},
    )
    ocr_provider: str = Field(
        default_factory=lambda: os.getenv("OCR_PROVIDER", "google"),
        description="OCR provider for non-digital PDFs (google, aws, azure)",
        json_schema_extra={"request_param": RequestParam.OCR_PROVIDER},
    )
    ocr_languages: List[str] = Field(
        default_factory=lambda: ["eng"],
        description="OCR languages",
        json_schema_extra={"request_param": RequestParam.OCR_LANGUAGES},
    )
    dpi: int = Field(
        default=300,
        description="DPI for image conversion",
        ge=150,
        le=600,
        json_schema_extra={"request_param": RequestParam.DPI},
    )
    extract_images: bool = Field(
        default=False,
        description="Extract embedded images",
        json_schema_extra={"request_param": RequestParam.EXTRACT_IMAGES},
    )


class ExcelParserConfig(ParserConfig):
    """Configuration for Excel parser."""

    sheet_names: Optional[List[str]] = Field(
        default=None,
        description="Sheets to parse (None = all sheets)",
        json_schema_extra={"request_param": RequestParam.SHEET_NAMES},
    )
    include_formulas: bool = Field(
        default=False,
        description="Include formula text",
        json_schema_extra={"request_param": RequestParam.INCLUDE_FORMULAS},
    )
    date_format: str = Field(
        default="%Y-%m-%d",
        description="Date format for cells",
        json_schema_extra={"request_param": RequestParam.DATE_FORMAT},
    )


class DocxParserConfig(ParserConfig):
    """Configuration for DOCX parser."""

    extract_styles: bool = Field(
        default=False,
        description="Extract style information",
        json_schema_extra={"request_param": RequestParam.EXTRACT_STYLES},
    )
    extract_comments: bool = Field(
        default=False,
        description="Extract document comments",
        json_schema_extra={"request_param": RequestParam.EXTRACT_COMMENTS},
    )
    preserve_formatting: bool = Field(
        default=True,
        description="Preserve text formatting",
        json_schema_extra={"request_param": RequestParam.PRESERVE_FORMATTING},
    )


class ImageParserConfig(ParserConfig):
    """Configuration for Image parser."""

    ocr_provider: str = Field(
        default_factory=lambda: os.getenv("OCR_PROVIDER", "google"),
        description="OCR provider (google, aws, azure)",
        json_schema_extra={"request_param": RequestParam.OCR_PROVIDER},
    )
    jpeg_quality: int = Field(
        default=85,
        description="JPEG quality for image compression before OCR (1-100)",
        ge=1,
        le=100,
        json_schema_extra={"request_param": RequestParam.JPEG_QUALITY},
    )
    fix_orientation: bool = Field(
        default=True,
        description="Automatically detect and correct image orientation",
        json_schema_extra={"request_param": RequestParam.ORIENTATION_CORRECTION},
    )
    dpi: int = Field(
        default=300,
        description="DPI for processing",
        ge=150,
        le=600,
        json_schema_extra={"request_param": RequestParam.DPI},
    )


class RequestConfig(BaseModel):
    """Configuration for parsing requests.

    Contains all parser-specific configurations that can be populated
    from request form data using the RequestConfigBuilder.
    """

    # All parser configs
    pdf_config: PDFParserConfig = Field(default_factory=PDFParserConfig)
    excel_config: ExcelParserConfig = Field(default_factory=ExcelParserConfig)
    docx_config: DocxParserConfig = Field(default_factory=DocxParserConfig)
    image_config: ImageParserConfig = Field(default_factory=ImageParserConfig)
