"""
Image document parser.

Uses OCR to extract text from image files.
Supports configurable OCR providers (Google Vision, AWS Textract, Azure Computer Vision).
"""

from pathlib import Path
from typing import Any, BinaryIO, Union

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from app.config.config import ImageParserConfig, ParserConfig
from app.utils import to_thread
from app.core.pdf.ocr.non_digital import (
    AWSTextractOCR,
    AzureComputerVisionOCR,
    GoogleVisionOCR,
)
from app.models.enums import ParsingType
from app.models.schemas import Block, Page, ParsedResponse
from app.services.parsers.base import Parser

# OCR provider mapping
OCR_PROVIDERS = {
    "google": GoogleVisionOCR,
    "aws": AWSTextractOCR,
    "azure": AzureComputerVisionOCR,
}


class ImageParser(Parser):
    """
    Parser for image files.

    Implements the 3-step parsing pipeline:
    - _pre_process: Loads image and runs OCR using selected provider
    - _parse: Stub for future parsing algorithm
    - _post_process: Creates ParsedResponse with OCR results

    Supports configurable OCR providers through the ImageParserConfig.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """
        Initialize the Image parser.

        Args:
            config: Parser configuration. If not provided, uses default ImageParserConfig.
        """
        if config is None:
            config = ImageParserConfig()
        elif not isinstance(config, ImageParserConfig):
            # Convert base config to Image config
            config = ImageParserConfig(**config.model_dump())
        super().__init__(config)

    @property
    def image_config(self) -> ImageParserConfig:
        """Get the Image-specific configuration."""
        return self.config  # type: ignore

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return [".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp", ".webp"]

    def _get_ocr_provider(self):
        """
        Get the OCR provider instance based on configuration.

        Returns:
            OCR provider instance (GoogleVisionOCR, AWSTextractOCR, or AzureComputerVisionOCR)

        Raises:
            ValueError: If the configured provider is not supported
        """
        provider_name = self.image_config.ocr_provider.lower()
        provider_class = OCR_PROVIDERS.get(provider_name)

        if provider_class is None:
            supported = ", ".join(OCR_PROVIDERS.keys())
            raise ValueError(f"Unsupported OCR provider '{provider_name}'. Supported: {supported}")

        logger.debug(f"Using OCR provider: {provider_name}")
        return provider_class(
            jpeg_quality=self.image_config.jpeg_quality,
            fix_orientation=self.image_config.fix_orientation,
        )

    async def _pre_process(self, file_path: Union[str, Path], file_obj: BinaryIO | None = None) -> dict[str, Any]:
        """
        Preprocess the image file.

        Loads the image and runs OCR using the configured provider.

        Args:
            file_path: Path to the image file
            file_obj: Optional file-like object

        Returns:
            Dictionary with standardized preprocessed data:
            - file_metadata: dict with page_scalar (list) and is_digital (None for images)
            - raw_ocr: List[pd.DataFrame] - OCR results (single item for images)
            - file_type: "image"
        """
        path = Path(file_path)
        logger.info(f"Pre-processing image: {path.name}")

        # Run OCR in thread pool to avoid blocking
        result = await to_thread(self._pre_process_sync, path)

        return result

    def _pre_process_sync(self, file_path: Path) -> dict[str, Any]:
        """
        Synchronous image preprocessing implementation.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with standardized preprocessed data:
            - file_metadata: dict with page_scalar (list) and is_digital (None)
            - raw_ocr: List[pd.DataFrame] - OCR results
            - file_type: "image"
        """
        # Load image to get dimensions
        image = Image.open(file_path)
        width, height = image.size

        # Convert to numpy array for OCR
        image_array = np.array(image.convert("RGB"))

        # Run OCR
        ocr = self._get_ocr_provider()
        logger.info(f"Running OCR on image: {file_path.name}")

        # Process single image
        ocr_df = ocr.process_image(image_array)

        return {
            "file_metadata": {
                "page_scalar": [{"width": width, "height": height, "angle": 0}],
                "is_digital": None,  # Not applicable for images
            },
            "raw_ocr": [ocr_df],  # Single DataFrame in list for consistency
            "file_type": "image",
        }

    async def _parse(self, preprocessed_data: dict[str, Any]) -> dict[str, Any]:
        """
        Parse the preprocessed image data.

        This is a stub for future parsing algorithm implementation.
        Currently returns the preprocessed data unchanged.

        Args:
            preprocessed_data: Data from _pre_process()

        Returns:
            Parsed data (currently unchanged preprocessed data)
        """
        # TODO: Implement parsing algorithm here
        # For now, return preprocessed data unchanged
        return preprocessed_data

    async def _post_process(self, parsed_data: dict[str, Any], file_path: Union[str, Path]) -> ParsedResponse:
        """
        Create ParsedResponse from parsed data.

        Args:
            parsed_data: Data from _parse() with standardized structure:
                - file_metadata: dict with page_scalar and is_digital
                - raw_ocr: List[pd.DataFrame]
                - file_type: "image"
            file_path: Original file path

        Returns:
            ParsedResponse with extracted content and metadata
        """
        path = Path(file_path)
        file_metadata = parsed_data["file_metadata"]
        raw_ocr = parsed_data["raw_ocr"]
        page_scalar = file_metadata["page_scalar"][0]  # Images have single page

        # Extract OCR DataFrame (single item for images)
        ocr_df = raw_ocr[0] if raw_ocr else pd.DataFrame()

        # Extract text from OCR DataFrame
        page_text = self._extract_text_from_df(ocr_df)

        # Create blocks from OCR DataFrame
        blocks = self._create_blocks_from_df(ocr_df, page_number=1, page_scalar=page_scalar)

        # Create single page (images are single-page documents)
        page = Page(
            page_number=1,
            page_content=page_text,
            blocks=blocks,
            content_bbox=[0, 0, page_scalar["width"], page_scalar["height"]],
        )

        # Build final file_metadata for response
        response_metadata = self._get_file_metadata(path)
        response_metadata["page_scalar"] = file_metadata["page_scalar"]
        response_metadata["is_digital"] = file_metadata["is_digital"]

        return ParsedResponse(
            content=page_text,
            file_type="img",
            file_metadata=response_metadata,
            parsing_type=ParsingType.OCR,
            pages=[page],
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
