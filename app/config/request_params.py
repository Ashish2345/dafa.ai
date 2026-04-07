"""
Request parameter enum module.

Contains the RequestParam enum for all available request parameters
that can be passed in form data to configure document parsing.
"""

from enum import StrEnum


class RequestParam(StrEnum):
    """Enum for all available request parameters."""

    # Base parser params
    EXTRACT_METADATA = "extract_metadata"
    EXTRACT_TABLES = "extract_tables"

    # PDF parser params
    OCR_ENABLED = "ocr_enabled"
    OCR_PROVIDER = "ocr_provider"
    OCR_LANGUAGES = "ocr_languages"
    DPI = "dpi"
    EXTRACT_IMAGES = "extract_images"

    # Image parser params
    JPEG_QUALITY = "jpeg_quality"
    ORIENTATION_CORRECTION = "orientation_correction"
