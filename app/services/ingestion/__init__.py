"""
Document Ingestion Service

ONLY gathers/collects data from parsed documents.
Does NOT perform processing - that's handled by ProcessingService.

This service is responsible for:
- Collecting OCR data from parsed responses
- Gathering page images and metadata
- Preparing raw data for processing
"""

from .service import IngestionService

__all__ = [
    "IngestionService",
]
