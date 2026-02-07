"""
Ingestion Service

ONLY gathers/collects data from parsed documents.
Does NOT perform any processing - that's handled by ProcessingService.

This service is responsible for:
- Collecting OCR data from parsed responses
- Gathering page images and metadata
- Preparing raw data for processing
"""

from typing import Any, Dict, List, Optional

from loguru import logger


class IngestionService:
    """
    Service for gathering/collecting document data for processing.

    This service ONLY gathers data - it does NOT process it.
    Processing is handled by ProcessingService.
    """

    def gather_document_data(
        self,
        parsed_response: Any,
        document_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Gather/collect document data from a parsed response.

        This method extracts all necessary data from the parsed response
        and prepares it for processing. It does NOT perform any processing.

        Args:
            parsed_response: ParsedResponse object from parser
            document_name: Optional document name/identifier

        Returns:
            Dictionary with gathered data:
            {
                "document_id": str,
                "raw_ocr": List[pd.DataFrame],  # OCR data per page
                "page_scalars": List[Dict],      # Page metadata
                "page_images": List[Any],        # Page images (if available)
                "parsed_ocr": Dict,              # Structured OCR results
                "document_metadata": Dict,       # Basic document metadata
                "status": str,                   # "success" or "failed"
            }
        """
        logger.info(f"Gathering data for document: {document_name or 'unnamed'}")

        if not hasattr(parsed_response, "file_metadata"):
            logger.error("Parsed response does not have file_metadata")
            return {
                "document_id": document_name or "unknown",
                "raw_ocr": [],
                "page_scalars": [],
                "page_images": [],
                "parsed_ocr": {},
                "document_metadata": {},
                "status": "failed",
                "error": "Invalid parsed response structure",
            }

        file_metadata = parsed_response.file_metadata
        parsed_ocr_data = file_metadata.get("parsed_ocr")

        if not parsed_ocr_data:
            is_digital = file_metadata.get("is_digital", False)
            error_msg = (
                "Document is digital (text-based). OCR data is not available."
                if is_digital
                else "No OCR data found. Ensure OCR is enabled (ocr_enabled=true)."
            )
            logger.warning(error_msg)
            return {
                "document_id": document_name or "unknown",
                "raw_ocr": [],
                "page_scalars": [],
                "page_images": [],
                "parsed_ocr": {},
                "document_metadata": {},
                "status": "failed",
                "error": error_msg,
            }

        # Extract cleaned_ocr from parsed_ocr_data
        cleaned_ocr_data = parsed_ocr_data.get("cleaned_ocr", [])

        # Convert to DataFrames if needed (handles both in-memory and serialized formats)
        import pandas as pd

        raw_ocr = []
        for page_data in cleaned_ocr_data:
            if isinstance(page_data, pd.DataFrame):
                raw_ocr.append(page_data)
            elif isinstance(page_data, list) and len(page_data) > 0:
                raw_ocr.append(pd.DataFrame(page_data))
            else:
                raw_ocr.append(pd.DataFrame())

        # Gather page metadata
        page_scalars = file_metadata.get("page_scalar", [])

        # Gather page images (if available)
        # TODO: Store page images during parsing for table extraction
        page_images = []

        # Gather basic document metadata
        document_metadata = {
            "document_id": document_name or "unknown",
            "file_type": getattr(parsed_response, "file_type", "pdf"),
            "parsing_type": str(getattr(parsed_response, "parsing_type", "")),
            "is_digital": file_metadata.get("is_digital", False),
            "total_pages": len(raw_ocr),
        }

        logger.info(
            f"Gathered data: {len(raw_ocr)} pages, "
            f"parsed_ocr keys: {list(parsed_ocr_data.keys()) if parsed_ocr_data else []}"
        )

        return {
            "document_id": document_name or "unknown",
            "raw_ocr": raw_ocr,
            "page_scalars": page_scalars,
            "page_images": page_images,
            "parsed_ocr": parsed_ocr_data,
            "document_metadata": document_metadata,
            "status": "success",
        }
