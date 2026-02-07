"""
PDF document parser.

Uses the existing core PDF reader infrastructure for parsing PDF documents.
Supports OCR for non-digital (scanned) PDFs with configurable OCR provider.
"""

from pathlib import Path
from typing import Any, BinaryIO, Union

import pandas as pd
from loguru import logger

from app.config.config import ParserConfig, PDFParserConfig
from app.utils import to_thread
from app.core.pdf.ocr.digital.ocr import DigitalOCR
from app.core.pdf.ocr.non_digital import (
    AWSTextractOCR,
    AzureComputerVisionOCR,
    GoogleVisionOCR,
)
from app.core.pdf.reader.reader import PDFReader
from app.models.enums import ParsingType
from app.models.schemas import Block, Page, ParsedResponse
from app.services.parsers.base import Parser
from app.services.parsers.ocr_input_parser import OCRInputParser

# OCR provider mapping
OCR_PROVIDERS = {
    "google": GoogleVisionOCR,
    "aws": AWSTextractOCR,
    "azure": AzureComputerVisionOCR,
}


class PDFParser(Parser):
    """
    Parser for PDF documents.

    Implements the 3-step parsing pipeline:
    - _pre_process: Reads PDF, detects if digital, runs OCR if needed
    - _parse: Stub for future parsing algorithm
    - _post_process: Creates ParsedResponse with extracted content

    Leverages the existing PDFReader infrastructure from app.core.pdf.reader
    to extract text and metadata from PDF files.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """
        Initialize the PDF parser.

        Args:
            config: Parser configuration. If not provided, uses default PDFParserConfig.
        """
        if config is None:
            config = PDFParserConfig()
        elif not isinstance(config, PDFParserConfig):
            # Convert base config to PDF config
            config = PDFParserConfig(**config.model_dump())
        super().__init__(config)

    @property
    def pdf_config(self) -> PDFParserConfig:
        """Get the PDF-specific configuration."""
        return self.config  # type: ignore

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return [".pdf"]

    def _get_ocr_provider(self):
        """
        Get the OCR provider instance based on configuration.

        Returns:
            OCR provider instance (GoogleVisionOCR, AWSTextractOCR, or AzureComputerVisionOCR)

        Raises:
            ValueError: If the configured provider is not supported
        """
        provider_name = self.pdf_config.ocr_provider.lower()
        provider_class = OCR_PROVIDERS.get(provider_name)

        if provider_class is None:
            supported = ", ".join(OCR_PROVIDERS.keys())
            raise ValueError(f"Unsupported OCR provider '{provider_name}'. Supported: {supported}")

        logger.debug(f"Using OCR provider: {provider_name}")
        
        # Pass language hints to OCR provider if available
        ocr_languages = getattr(self.pdf_config, "ocr_languages", None)
        if ocr_languages and provider_name == "google":
            logger.info(f"Configuring OCR with languages: {ocr_languages}")
            return provider_class(language_hints=ocr_languages)
        
        return provider_class()

    async def _pre_process(self, file_path: Union[str, Path], file_obj: BinaryIO | None = None) -> dict[str, Any]:
        """
        Preprocess the PDF file.

        Reads the PDF, detects if it's digital (has extractable text),
        and runs OCR if the PDF is non-digital and OCR is enabled.

        Args:
            file_path: Path to the PDF file
            file_obj: Optional file-like object (not used for PDF)

        Returns:
            Dictionary containing:
            - pages_data: List of page data with text and dimensions
            - is_digital: Whether the PDF has extractable text
            - parsing_type: ParsingType used (TEXT or OCR)
            - metadata: PDF metadata (page_count, dpi)
        """
        path = Path(file_path)
        logger.info(f"Pre-processing PDF: {path.name}")

        # Run PDF reading in thread pool to avoid blocking
        result = await to_thread(self._pre_process_sync, path)

        return result

    def _pre_process_sync(self, file_path: Path) -> dict[str, Any]:
        """
        Synchronous PDF preprocessing implementation.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with standardized preprocessed data:
            - file_metadata: dict with page_scalar (list) and is_digital (bool)
            - raw_ocr: List[pd.DataFrame] - OCR results per page
            - file_type: "pdf"
        """
        page_scalars: list[dict[str, Any]] = []
        total_words = 0

        with PDFReader(
            str(file_path),
            dpi=self.pdf_config.dpi,
            use_dynamic_dpi=True,
        ) as reader:
            # First pass: check if PDF has extractable text and collect page scalars
            for pdf_page in reader:
                data = pdf_page.data
                words = data.words if data else []
                total_words += len(words) if words else 0

                # Collect page scalar as dict
                page_scalars.append(data.page_scalar.to_dict())

        # Determine if PDF is digital (has extractable text)
        # Consider non-digital if average words per page is very low
        avg_words_per_page = total_words / len(page_scalars) if page_scalars else 0
        is_digital = avg_words_per_page > 10  # Threshold for digital detection

        # Always run OCR - DigitalOCR for digital PDFs, non-digital OCR for scanned
        if is_digital:
            logger.info(f"PDF is digital (avg {avg_words_per_page:.1f} words/page). Running DigitalOCR...")
            raw_ocr = self._run_ocr(file_path, is_digital=True)
        else:
            if self.pdf_config.ocr_enabled:
                logger.info(f"PDF appears to be non-digital (avg {avg_words_per_page:.1f} words/page). Running OCR...")
                raw_ocr = self._run_ocr(file_path, is_digital=False)
            else:
                logger.info("OCR disabled, returning empty OCR results")
                raw_ocr = [pd.DataFrame() for _ in page_scalars]

        # Get page images for AWS table extraction
        page_images = []
        try:
            with PDFReader(
                str(file_path),
                dpi=self.pdf_config.dpi,
                use_dynamic_dpi=True,
            ) as reader:
                for page_idx in range(min(len(page_scalars), reader.page_count)):
                    page = reader[page_idx]
                    image = page.to_image()
                    page_images.append(image)
        except Exception as e:
            logger.warning(f"Error getting page images for table extraction: {e}")

        return {
            "file_metadata": {
                "page_scalar": page_scalars,
                "is_digital": is_digital,
            },
            "raw_ocr": raw_ocr,
            "page_images": page_images,
            "file_type": "pdf",
        }

    def _run_ocr(self, file_path: Path, is_digital: bool = False) -> list[pd.DataFrame]:
        """
        Run OCR on the PDF using the appropriate provider.

        Args:
            file_path: Path to the PDF file
            is_digital: If True, use DigitalOCR; otherwise use configured non-digital OCR provider

        Returns:
            List of DataFrames with OCR results, one per page
        """
        if is_digital:
            ocr = DigitalOCR()
            logger.debug("Using DigitalOCR for digital PDF")
        else:
            ocr = self._get_ocr_provider()
            logger.debug(f"Using {self.pdf_config.ocr_provider} OCR for non-digital PDF")

        with PDFReader(
            str(file_path),
            dpi=self.pdf_config.dpi,
            use_dynamic_dpi=True,
        ) as reader:
            # Process document with OCR
            ocr_results = ocr.process_document(reader)

        return ocr_results

    async def _parse(self, preprocessed_data: dict[str, Any]) -> dict[str, Any]:
        """
        Parse the preprocessed PDF data using the OCR parser component.

        This method uses the OCRParser to:
        - Clean and normalize OCR data
        - Remove noise and inconsistencies
        - Group fields, tables, and multi-page content
        - Produce a deterministic, agent-friendly schema

        Args:
            preprocessed_data: Data from _pre_process() containing:
                - file_metadata: dict with page_scalar and is_digital
                - raw_ocr: List[pd.DataFrame] - OCR results per page
                - file_type: "pdf"

        Returns:
            Parsed data with structured OCR parsing results:
                - file_metadata: Original metadata
                - parsed_ocr: Structured parsing results from OCRParser
                - file_type: "pdf"
        """
        logger.info("Parsing OCR data with OCR parser component")

        # Extract data from preprocessed input
        file_metadata = preprocessed_data.get("file_metadata", {})
        raw_ocr = preprocessed_data.get("raw_ocr", [])
        page_scalars = file_metadata.get("page_scalar", [])
        page_images = preprocessed_data.get("page_images", [])

        # Initialize OCR input parser
        ocr_parser = OCRInputParser(
            min_confidence=0.5,
            enable_table_detection=False,  # Disabled by default
            enable_field_grouping=True,
            combine_words=True,
        )

        # Run parsing in thread pool to avoid blocking
        parsed_result = await to_thread(
            ocr_parser.parse,
            raw_ocr=raw_ocr,
            page_scalars=page_scalars if page_scalars else None,
            page_images=page_images,
        )

        # Merge parsed results with original metadata
        return {
            "file_metadata": file_metadata,
            "parsed_ocr": parsed_result,
            "raw_ocr": raw_ocr,  # Keep original for backward compatibility
            "file_type": preprocessed_data.get("file_type", "pdf"),
        }

    async def _post_process(self, parsed_data: dict[str, Any], file_path: Union[str, Path]) -> ParsedResponse:
        """
        Create ParsedResponse from parsed data.

        Uses the structured OCR parsing results if available, otherwise falls back
        to direct OCR DataFrame processing.

        Args:
            parsed_data: Data from _parse() with structure:
                - file_metadata: dict with page_scalar and is_digital
                - parsed_ocr: Structured parsing results (optional)
                - raw_ocr: List[pd.DataFrame] - OCR results per page
                - file_type: "pdf"
            file_path: Original file path

        Returns:
            ParsedResponse with extracted content and metadata
        """
        path = Path(file_path)
        file_metadata = parsed_data["file_metadata"]
        page_scalars = file_metadata["page_scalar"]
        is_digital = file_metadata["is_digital"]

        # Determine parsing type based on is_digital
        parsing_type = ParsingType.TEXT if is_digital else ParsingType.OCR

        # Use parsed OCR results if available, otherwise use raw OCR
        parsed_ocr = parsed_data.get("parsed_ocr")
        raw_ocr = parsed_data.get("raw_ocr", [])

        pages: list[Page] = []
        all_content: list[str] = []

        if parsed_ocr and parsed_ocr.get("pages"):
            # Use structured parsed results
            logger.debug("Using structured OCR parsing results")
            for page_data in parsed_ocr["pages"]:
                page_number = page_data.get("page_number", 0)
                page_content = page_data.get("content", "")
                page_blocks_data = page_data.get("blocks", [])
                page_tables = page_data.get("tables", [])
                page_bbox = page_data.get("bbox", [0, 0, 0, 0])

                # Convert block data to Block objects
                blocks = []
                for block_data in page_blocks_data:
                    blocks.append(
                        Block(
                            block_id=block_data.get("block_id", f"block_{page_number}_0"),
                            block_parsed_content=block_data.get("content", ""),
                            block_bbox=block_data.get("bbox", [0, 0, 0, 0]),
                        )
                    )

                # Add table blocks as HTML
                for table_data in page_tables:
                    table_html = table_data.get("html", "")
                    if table_html:
                        blocks.append(
                            Block(
                                block_id=table_data.get("table_id", f"table_{page_number}_0"),
                                block_parsed_content=table_html,
                                block_raw_content=table_html,
                                block_bbox=table_data.get("bbox", [0, 0, 0, 0]),
                                block_html_tags={"type": "table"},
                            )
                        )

                # Get page scalar for this page
                page_scalar = (
                    page_scalars[page_number - 1] if page_number <= len(page_scalars) else {}
                )

                page = Page(
                    page_number=page_number,
                    page_content=page_content,
                    blocks=blocks,
                    content_bbox=page_bbox if page_bbox else [0, 0, page_scalar.get("width", 0), page_scalar.get("height", 0)],
                )
                pages.append(page)
                all_content.append(page_content)

            full_content = parsed_ocr.get("content", "\n\n".join(all_content))
        else:
            # Fallback to direct OCR DataFrame processing
            logger.debug("Using direct OCR DataFrame processing (fallback)")
            for page_idx, (ocr_df, page_scalar) in enumerate(zip(raw_ocr, page_scalars)):
                page_number = page_idx + 1

                # Extract text and blocks from OCR DataFrame
                page_text = self._extract_text_from_df(ocr_df)
                blocks = self._create_blocks_from_df(ocr_df, page_number, page_scalar)

                all_content.append(page_text)

                page = Page(
                    page_number=page_number,
                    page_content=page_text,
                    blocks=blocks,
                    content_bbox=[0, 0, page_scalar["width"], page_scalar["height"]],
                )
                pages.append(page)

            full_content = "\n\n".join(all_content)

        # Build final file_metadata for response
        response_metadata = self._get_file_metadata(path)
        response_metadata["page_scalar"] = page_scalars
        response_metadata["is_digital"] = is_digital

        # Add parsing metadata if available
        if parsed_ocr and parsed_ocr.get("metadata"):
            response_metadata["parsing_metadata"] = parsed_ocr["metadata"]
            response_metadata["tables_detected"] = parsed_ocr.get("tables", [])
            response_metadata["fields_detected"] = parsed_ocr.get("fields", {})

        # Store parsed_ocr and raw_ocr in file_metadata for ingestion service
        # This allows the ingest endpoint to access the full OCR data
        if parsed_ocr:
            response_metadata["parsed_ocr"] = parsed_ocr
        if raw_ocr:
            # Store raw_ocr as list of DataFrames (will be serialized as dicts in JSON)
            # For ingestion, we need to convert back to DataFrames
            response_metadata["raw_ocr_dataframes"] = [
                df.to_dict("records") if df is not None and not df.empty else []
                for df in raw_ocr
            ]

        return ParsedResponse(
            content=full_content,
            file_type="pdf",
            file_metadata=response_metadata,
            parsing_type=parsing_type,
            pages=pages,
        )

    def _extract_text_from_df(self, ocr_df: pd.DataFrame) -> str:
        """
        Extract text from OCR DataFrame.

        Args:
            ocr_df: DataFrame with OCR results (columns: Text, x0, y0, x2, y2, block, line, etc.)

        Returns:
            Extracted text as string
        """
        if ocr_df is None or ocr_df.empty:
            return ""

        # Group words by block and line
        lines: dict[tuple[int, int], list] = {}
        for _, row in ocr_df.iterrows():
            key = (int(row.get("block", 0)), int(row.get("line", 0)))
            if key not in lines:
                lines[key] = []
            lines[key].append({"text": row.get("Text", ""), "x0": row.get("x0", 0)})

        # Sort and join words
        text_parts = []
        for key in sorted(lines.keys()):
            line_words = sorted(lines[key], key=lambda w: w["x0"])
            line_text = " ".join(str(w["text"]) for w in line_words if w["text"])
            if line_text:
                text_parts.append(line_text)

        return "\n".join(text_parts)

    def _create_blocks_from_df(
        self, ocr_df: pd.DataFrame, page_number: int, page_scalar: dict[str, Any]
    ) -> list[Block]:
        """
        Create Block objects from OCR DataFrame.

        Args:
            ocr_df: DataFrame with OCR results
            page_number: Page number for block IDs
            page_scalar: Page dimensions dict with width, height, angle

        Returns:
            List of Block objects
        """
        if ocr_df is None or ocr_df.empty:
            return []

        # Group rows by block
        block_rows: dict[int, list[dict]] = {}
        for _, row in ocr_df.iterrows():
            block_id = int(row.get("block", 0))
            if block_id not in block_rows:
                block_rows[block_id] = []
            block_rows[block_id].append(
                {
                    "text": row.get("Text", ""),
                    "x0": row.get("x0", 0),
                    "y0": row.get("y0", 0),
                    "x2": row.get("x2", 0),
                    "y2": row.get("y2", 0),
                    "line": int(row.get("line", 0)),
                }
            )

        blocks = []
        page_width = page_scalar.get("width", 1)
        page_height = page_scalar.get("height", 1)

        for block_id, rows in sorted(block_rows.items()):
            # Calculate block bounding box (convert from normalized to pixel coords)
            x0 = min(r["x0"] for r in rows) * page_width
            y0 = min(r["y0"] for r in rows) * page_height
            x2 = max(r["x2"] for r in rows) * page_width
            y2 = max(r["y2"] for r in rows) * page_height

            # Group by line and extract text
            lines: dict[int, list] = {}
            for r in rows:
                line_num = r["line"]
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append(r)

            text_parts = []
            for line_num in sorted(lines.keys()):
                line_words = sorted(lines[line_num], key=lambda w: w["x0"])
                line_text = " ".join(str(w["text"]) for w in line_words if w["text"])
                if line_text:
                    text_parts.append(line_text)

            block_text = "\n".join(text_parts)

            blocks.append(
                Block(
                    block_id=f"block_{page_number}_{block_id}",
                    block_parsed_content=block_text,
                    block_bbox=[x0, y0, x2, y2],
                )
            )

        return blocks

