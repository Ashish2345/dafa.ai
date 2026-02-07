"""
Form data models for API endpoints.

Contains Pydantic models with as_form() dependencies for handling
multipart form data with file uploads.
"""

import inspect
from typing import List, Literal, Optional

from fastapi import Form
from pydantic import BaseModel, Field


class ParseFormData(BaseModel):
    """
    Form data model for document parsing endpoint.

    Contains all configurable parameters for parsing different document types.
    Use with Depends(ParseFormData.as_form()) to parse multipart form data.
    """

    # File input params
    file_type: Literal["url", "file"] = Field("file", description="Input type: 'file' for upload, 'url' for signed URL")
    file_url: Optional[str] = Field(None, description="URL to download file from (when file_type='url')")
    file_path: Optional[str] = Field(None, exclude=True, description="Path to saved file (populated after processing)")

    # Common params
    dpi: Optional[int] = Field(None, description="DPI for image/PDF processing (150-600)")
    extract_metadata: Optional[bool] = Field(None, description="Extract document metadata")
    extract_tables: Optional[bool] = Field(None, description="Extract tables from document")

    # OCR params
    ocr_enabled: Optional[bool] = Field(None, description="Enable OCR for scanned PDFs")
    ocr_provider: Optional[str] = Field(None, description="OCR provider (google, aws, azure)")
    ocr_languages: Optional[List[str]] = Field(None, description="OCR languages")

    # PDF-specific params
    extract_images: Optional[bool] = Field(None, description="Extract embedded images from PDF")

    # Image-specific params
    orientation_correction: Optional[bool] = Field(None, description="Auto-detect and correct image orientation")
    jpeg_quality: Optional[int] = Field(None, description="JPEG quality for image compression (1-100)")

    # Excel-specific params
    sheet_names: Optional[List[str]] = Field(None, description="Excel sheets to parse (None = all)")
    include_formulas: Optional[bool] = Field(None, description="Include Excel formula text")
    date_format: Optional[str] = Field(None, description="Date format for Excel cells")

    # Docx-specific params
    extract_styles: Optional[bool] = Field(None, description="Extract DOCX style information")
    extract_comments: Optional[bool] = Field(None, description="Extract DOCX document comments")
    preserve_formatting: Optional[bool] = Field(None, description="Preserve DOCX text formatting")

    @classmethod
    def as_form(cls):
        """Create a FastAPI dependency for parsing multipart form data."""
        new_params = []

        # Fields that are not form inputs (populated after processing)
        excluded_fields = {"file_path"}

        for field_name, model_field in cls.model_fields.items():
            if field_name in excluded_fields:
                continue

            default = model_field.default
            description = model_field.description
            form_default = Form(default, description=description)

            new_params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=form_default,
                    annotation=model_field.annotation,
                )
            )

        async def _as_form_func(**data) -> "ParseFormData":
            return cls(**data)

        sig = inspect.signature(_as_form_func)
        sig = sig.replace(parameters=new_params)
        _as_form_func.__signature__ = sig

        return _as_form_func
