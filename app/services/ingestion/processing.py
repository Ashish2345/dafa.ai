"""
Document processing — OCR to Markdown and metadata extraction.
Re-exports from existing modules during migration.
"""

from app.services.processing.markdown import DocumentProcessor
from app.services.processing.metadata import MetadataExtractor

__all__ = ["DocumentProcessor", "MetadataExtractor"]
