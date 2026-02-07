"""
Digital OCR module for extracting text from digitally-created PDFs.

Digital PDFs contain embedded text that can be extracted directly without
running optical character recognition. This module provides the DigitalOCR
class that leverages the PDF reader's text extraction capabilities.
"""

from .ocr import DigitalOCR

__all__ = ["DigitalOCR"]
