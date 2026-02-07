"""
AWS Textract OCR implementation for non-digital (scanned) documents.

This module provides both synchronous and asynchronous AWS Textract OCR
implementations with support for multiple image sources.

Example:
    Synchronous usage:
        from diu_new.ocr import AWSTextractOCR
        from diu_new.pdf_reader import PDFReader

        ocr = AWSTextractOCR(jpeg_quality=85, fix_orientation=True)

        # Process a PDF document
        with PDFReader("document.pdf") as reader:
            df_list = ocr.process_document(reader)

        # Or process a single image
        df = ocr.process_image("image.jpg")

    Asynchronous usage:
        from diu_new.ocr import AWSTextractOCRAsync

        async_ocr = AWSTextractOCRAsync()
        df = await async_ocr.process_image_async(image_array)
"""

import asyncio
import io
from abc import abstractmethod
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import aioboto3
import boto3
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from app.utils import to_thread

from ..base import OCR_COLUMNS, OCRBase
from ..utils import (
    get_orientation_angle_from_df,
    get_rotated_df_4point,
    remove_no_break_space,
    sort_df,
)

if TYPE_CHECKING:
    from ...reader import PDFPage, PDFReader

# Type alias for supported image inputs
ImageInput = Union[bytes, str, np.ndarray, Path, Image.Image]

# Type alias for AWS Textract response
TextractResponse = Dict[str, Any]


class AWSTextractOCRBase(OCRBase):
    """
    Abstract base class for AWS Textract OCR implementations.

    This class provides common functionality for both sync and async
    implementations, including image construction, response parsing,
    and orientation correction.

    Args:
        jpeg_quality: int
            JPEG encoding quality (1-100) for image compression before API call.
            Lower values reduce memory and network usage. Default: 85
        fix_orientation: bool
            If True, automatically detect and correct document orientation.
            Default: True
        reset_lines_and_sort: bool
            If True, recalculates line assignments and sorts the DataFrame.
            Default: True
        max_workers: int, optional
            Maximum number of threads for parallel page processing.
            If None, uses ThreadPoolExecutor default (min(32, cpu_count + 4)).
            Default: None
    """

    def __init__(
        self,
        jpeg_quality: int = 85,
        fix_orientation: bool = True,
        reset_lines_and_sort: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.jpeg_quality = jpeg_quality
        self.fix_orientation = fix_orientation
        self.reset_lines_and_sort = reset_lines_and_sort
        self.max_workers = max_workers

    def _construct_textract_document(self, image: ImageInput) -> bytes:
        """
        Construct bytes for AWS Textract Document from various types of input.

        Supports: bytes, base64 string, numpy array, file path (including PDF),
        and PIL Image.

        Note: AWS Textract does not support URL input directly, so URLs are not
        supported in this implementation.

        Args:
            image: The input image in any supported format

        Returns:
            bytes ready for AWS Textract API call

        Raises:
            ValueError: If image type is not supported
        """
        # Handle bytes directly
        if isinstance(image, bytes):
            return image

        # Handle base64 string
        if isinstance(image, str):
            # Check if it's a base64 string (ends with = and no spaces)
            if image.endswith("=") and " " not in image:
                try:
                    return b64decode(bytes(image, encoding="utf8"))
                except Exception:
                    pass

            # Treat as file path
            image = Path(image)

        # Handle Path objects
        if isinstance(image, Path):
            if not image.exists():
                raise ValueError(f"File not found: {image}")

            ext = image.suffix.lower()

            # Handle PDF files (extract first page)
            if ext == ".pdf":
                # Import here to avoid circular imports
                from ...reader import PDFReader

                with PDFReader(str(image)) as reader:
                    if reader.page_count == 0:
                        raise ValueError("PDF has no pages")
                    page_image = reader[0].to_image()

                # Encode as JPEG with quality setting
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                _, buffer = cv2.imencode(".jpg", page_image, encode_params)
                content = buffer.tobytes()
                del buffer  # Free memory
                return content

            # Handle image files
            if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}:
                with open(image, "rb") as f:
                    return f.read()

            raise ValueError(f"Unsupported file format: {ext}")

        # Handle numpy array (BGR format from OpenCV)
        if isinstance(image, np.ndarray):
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, buffer = cv2.imencode(".jpg", image, encode_params)
            if not success:
                raise ValueError("Failed to encode numpy array as JPEG")
            content = buffer.tobytes()
            del buffer  # Free memory immediately
            return content

        # Handle PIL Image
        if isinstance(image, Image.Image):
            with io.BytesIO() as buffer:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(buffer, format="JPEG", quality=self.jpeg_quality)
                return buffer.getvalue()

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _parse_response_to_df(
        self,
        response: TextractResponse,
        width: int,
        height: int,
        page_index: int = 0,
    ) -> pd.DataFrame:
        """
        Parse AWS Textract response to a DataFrame.

        Extracts word-level data with 4-point coordinates for accurate
        orientation correction.

        AWS Textract polygon ordering:
            0----------1
            | TEXT     |
            3----------2

        Args:
            response: AWS Textract API response
            width: Image width in pixels (for coordinate scaling)
            height: Image height in pixels (for coordinate scaling)
            page_index: Page index for the 'page' column

        Returns:
            pd.DataFrame with OCR columns plus 4-point coordinates
        """
        data = []

        blocks = response.get("Blocks", [])
        if not blocks:
            logger.warning("No blocks found in Textract response")
            return self.create_empty_df()

        # Build block lookup by ID for efficient traversal
        block_by_id = {block["Id"]: block for block in blocks}

        # Find PAGE blocks
        pages = [block for block in blocks if block["BlockType"] == "PAGE"]

        for page in pages:
            # Get LINE block IDs from page relationships
            line_ids = []
            for rel in page.get("Relationships", []):
                if rel["Type"] == "CHILD":
                    line_ids.extend(rel["Ids"])

            # Process each LINE
            for line_idx, line_id in enumerate(line_ids):
                line_block = block_by_id.get(line_id)
                if not line_block or line_block["BlockType"] != "LINE":
                    continue

                # Get WORD block IDs from line relationships
                word_ids = []
                for rel in line_block.get("Relationships", []):
                    if rel["Type"] == "CHILD":
                        word_ids.extend(rel["Ids"])

                # Process each WORD
                for word_idx, word_id in enumerate(word_ids):
                    word_block = block_by_id.get(word_id)
                    if not word_block or word_block["BlockType"] != "WORD":
                        continue

                    # Extract word text
                    word_text = word_block.get("Text", "")
                    word_text = remove_no_break_space(word_text)

                    if not word_text:
                        continue

                    # Get confidence (AWS returns 0-100, normalize to 0-1)
                    confidence = word_block.get("Confidence", 99.0) / 100.0

                    # Get bounding box (normalized coordinates)
                    geometry = word_block.get("Geometry", {})
                    bbox = geometry.get("BoundingBox", {})
                    polygon = geometry.get("Polygon", [])

                    # Calculate bounding box from normalized coordinates
                    x0_norm = bbox.get("Left", 0)
                    y0_norm = bbox.get("Top", 0)
                    w_norm = bbox.get("Width", 0)
                    h_norm = bbox.get("Height", 0)

                    # Scale to pixel coordinates
                    x0 = int(round(x0_norm * width))
                    y0 = int(round(y0_norm * height))
                    x2 = int(round((x0_norm + w_norm) * width))
                    y2 = int(round((y0_norm + h_norm) * height))

                    # Skip invalid boxes
                    if x0 >= x2 or y0 >= y2:
                        continue

                    # Extract 4-point polygon coordinates
                    # AWS polygon ordering: TL(0), TR(1), BR(2), BL(3)
                    if len(polygon) >= 4:
                        point_x0 = int(round(polygon[0].get("X", 0) * width))
                        point_y0 = int(round(polygon[0].get("Y", 0) * height))
                        point_x1 = int(round(polygon[1].get("X", 0) * width))
                        point_y1 = int(round(polygon[1].get("Y", 0) * height))
                        point_x2 = int(round(polygon[2].get("X", 0) * width))
                        point_y2 = int(round(polygon[2].get("Y", 0) * height))
                        point_x3 = int(round(polygon[3].get("X", 0) * width))
                        point_y3 = int(round(polygon[3].get("Y", 0) * height))
                    else:
                        # Fallback to bounding box corners
                        point_x0, point_y0 = x0, y0
                        point_x1, point_y1 = x2, y0
                        point_x2, point_y2 = x2, y2
                        point_x3, point_y3 = x0, y2

                    # Determine space_type based on next word
                    space_type = 1  # Default: space after word
                    try:
                        next_word_id = word_ids[word_idx + 1]
                        next_word = block_by_id.get(next_word_id)
                        if next_word and next_word.get("Text", "").startswith(":"):
                            space_type = 2  # No space before colon
                    except IndexError:
                        space_type = 3  # End of line

                    temp_data = {
                        "Text": word_text,
                        "x0": x0,
                        "y0": y0,
                        "x2": x2,
                        "y2": y2,
                        "block": 0,  # AWS doesn't provide block info like Google
                        "line": line_idx,
                        "space_type": space_type,
                        "page": page_index,
                        "confidence": confidence,
                        "index_sort": len(data),
                        # Store 4-point coordinates for accurate rotation
                        "point_x0": point_x0,
                        "point_y0": point_y0,
                        "point_x1": point_x1,
                        "point_y1": point_y1,
                        "point_x2": point_x2,
                        "point_y2": point_y2,
                        "point_x3": point_x3,
                        "point_y3": point_y3,
                    }

                    data.append(temp_data)

        if not data:
            return self.create_empty_df()

        df = pd.DataFrame(data)
        return df

    def _apply_orientation_correction(
        self,
        image: np.ndarray,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """
        Apply orientation correction to image and DataFrame.

        Uses the 4-point rotation method for accurate correction.
        Angle is calculated from the DataFrame's 4-point coordinates.

        Args:
            image: Original image array
            df: DataFrame with OCR data and 4-point coordinates

        Returns:
            Tuple of (corrected_df, rotated_image, angle)
        """
        if df.empty:
            return df, image, 0.0

        angle = get_orientation_angle_from_df(df)
        logger.debug(f"Detected orientation angle: {angle}")

        if abs(angle) < 0.5:
            # No significant rotation needed
            return df, image, 0.0

        # Check if 4-point columns exist
        point_cols = [f"point_{axis}{i}" for axis in ["x", "y"] for i in range(4)]
        has_4_points = all(col in df.columns for col in point_cols)

        if not has_4_points:
            logger.warning("4-point columns missing, skipping orientation correction")
            return df, image, 0.0

        try:
            logger.debug("Using 4-point rotation transformation")
            df, image_rotated = get_rotated_df_4point(image, df, angle)
        except Exception as e:
            logger.warning(f"Orientation correction failed: {e}")
            return df, image, 0.0

        return df, image_rotated, angle

    def _wordify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DataFrame to handle colon splitting and word combining.

        This handles cases where colons should be separated from adjacent words.

        Args:
            df: pandas.DataFrame with word-level OCR data

        Returns:
            Processed DataFrame with proper word boundaries
        """
        if df.empty:
            return df

        data = []
        rows = list(df.sort_index().itertuples())

        for i, row in enumerate(rows):
            text = row.Text
            x0, y0, x2, y2 = row.x0, row.y0, row.x2, row.y2
            space_type = row.space_type

            # Check if text contains colons that should be split
            # (colons preceded by non-digits or at start)
            import re

            matches = list(re.finditer(r"(?<=\D):|^:", text))

            if not matches:
                # No colons to split, keep word as-is
                data.append(
                    {
                        "index_sort": len(data),
                        "Text": text,
                        "x0": x0,
                        "y0": y0,
                        "x2": x2,
                        "y2": y2,
                        "space_type": space_type,
                        "line": row.line,
                        "block": row.block,
                        "page": row.page,
                        "confidence": row.confidence,
                    }
                )
                continue

            # Split on colons
            charwidth = (x2 - x0) / len(text) if len(text) > 0 else 0

            for j, match in enumerate(matches):
                start, end = match.start(), match.end()

                # Find end of token
                try:
                    eot = matches[j + 1].start()
                    space = 2
                except (IndexError, ValueError):
                    eot = len(text) - 1
                    space = space_type

                # Text before first colon
                if j == 0 and start > 0:
                    partial_text = text[:start]
                    partial_x0 = x0
                    partial_x2 = partial_x0 + charwidth * start
                    data.append(
                        {
                            "index_sort": len(data),
                            "Text": partial_text,
                            "x0": int(partial_x0),
                            "y0": y0,
                            "x2": int(partial_x2),
                            "y2": y2,
                            "space_type": 2,
                            "line": row.line,
                            "block": row.block,
                            "page": row.page,
                            "confidence": row.confidence,
                        }
                    )

                # The colon itself
                partial_x0 = x0 + charwidth * start
                partial_x2 = partial_x0 + charwidth
                data.append(
                    {
                        "index_sort": len(data),
                        "Text": ":",
                        "x0": int(partial_x0),
                        "y0": y0,
                        "x2": int(partial_x2),
                        "y2": y2,
                        "space_type": 2,
                        "line": row.line,
                        "block": row.block,
                        "page": row.page,
                        "confidence": row.confidence,
                    }
                )

                # Text after colon
                if end < eot:
                    partial_text = text[end : eot + 1]
                    partial_x0 = x0 + charwidth * end
                    partial_x2 = partial_x0 + charwidth * len(partial_text)
                    data.append(
                        {
                            "index_sort": len(data),
                            "Text": partial_text,
                            "x0": int(partial_x0),
                            "y0": y0,
                            "x2": int(partial_x2),
                            "y2": y2,
                            "space_type": space,
                            "line": row.line,
                            "block": row.block,
                            "page": row.page,
                            "confidence": row.confidence,
                        }
                    )

        if not data:
            return self.create_empty_df()

        result_df = pd.DataFrame(data, columns=OCR_COLUMNS)

        # Filter invalid boxes
        result_df = result_df[(result_df["x0"] < result_df["x2"]) & (result_df["y0"] < result_df["y2"])]
        result_df["index_sort"] = range(len(result_df))

        return result_df

    @abstractmethod
    def read_raw(self, image: ImageInput) -> TextractResponse:
        """Get raw AWS Textract response. Must be implemented by subclass."""
        pass


class AWSTextractOCR(AWSTextractOCRBase):
    """
    Synchronous AWS Textract OCR implementation.

    This class provides synchronous methods for OCR processing using
    AWS Textract API.

    Example:
        ocr = AWSTextractOCR(jpeg_quality=85)

        # Process a PDF document
        with PDFReader("document.pdf") as reader:
            df_list = ocr.process_document(reader)

        # Process a single image
        df = ocr.process_image("image.jpg")

        # Get raw response
        response = ocr.read_raw(image_array)
    """

    def read_raw(self, image: ImageInput) -> TextractResponse:
        """
        Get raw AWS Textract OCR response.

        Args:
            image: Input image in any supported format

        Returns:
            Dict response from AWS Textract API
        """
        logger.info("Using AWS Textract OCR provider (sync)")
        document_bytes = self._construct_textract_document(image)
        client = boto3.client("textract")
        return client.detect_document_text(Document={"Bytes": document_bytes})

    def process_image(
        self,
        image: ImageInput,
        page_index: int = 0,
    ) -> pd.DataFrame:
        """
        Process an image and extract OCR data.

        Supports: bytes, base64 string, numpy array, file path, PIL Image.

        Args:
            image: Input image in any supported format
            page_index: Page index for the 'page' column (default: 0)

        Returns:
            pd.DataFrame with OCR data
        """
        logger.info("Processing image with AWS Textract OCR")

        # Get image dimensions
        image_array = self._get_image_array(image)
        if image_array is None:
            raise ValueError("Could not get image dimensions")

        height, width = image_array.shape[:2]

        # Get raw response
        response = self.read_raw(image)

        # Parse response to DataFrame
        df = self._parse_response_to_df(response, width, height, page_index)

        if df.empty:
            return df

        # Apply orientation correction if enabled
        if self.fix_orientation:
            df, _, _ = self._apply_orientation_correction(image_array, df)

        # Apply wordify to handle colon splitting
        df = self._wordify(df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = sort_df(df)

        # Free memory
        del image_array

        return df[OCR_COLUMNS] if not df.empty else df

    def process_page(self, page: "PDFPage", page_index: int = 0) -> pd.DataFrame:
        """
        Process a single PDF page and extract OCR data.

        Args:
            page: PDFPage object to process
            page_index: Zero-based page index for the 'page' column

        Returns:
            pd.DataFrame with standardized OCR columns
        """
        logger.info(f"Processing page {page_index} with AWS Textract OCR")

        # Get page image (lazy loaded)
        image = page.to_image()
        height, width = image.shape[:2]

        # Get raw response
        response = self.read_raw(image)

        # Parse response to DataFrame
        df = self._parse_response_to_df(response, width, height, page_index)

        if df.empty:
            logger.debug(f"No text found on page {page_index}")
            return self.create_empty_df()

        # Apply orientation correction if enabled
        if self.fix_orientation:
            df, _, _ = self._apply_orientation_correction(image, df)

        # Apply wordify to handle colon splitting
        df = self._wordify(df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = sort_df(df)

        # Free the image memory
        del image

        return df[OCR_COLUMNS] if not df.empty else df

    def process_document(self, reader: "PDFReader") -> List[pd.DataFrame]:
        """
        Process all pages in a PDF document in parallel.

        Pages are processed concurrently using ThreadPoolExecutor for better
        performance on multi-page documents.

        Args:
            reader: PDFReader instance

        Returns:
            List of DataFrames, one per page (in page order)
        """
        logger.info(f"Processing document with AWS Textract OCR ({reader.page_count} pages)")

        # Collect pages with their indices
        pages_with_indices = list(enumerate(reader))

        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.process_page, page, i): i for i, page in pages_with_indices}

            # Collect results maintaining page order
            results: List[pd.DataFrame] = [None] * len(pages_with_indices)  # type: ignore
            for future in as_completed(futures):
                page_idx = futures[future]
                results[page_idx] = future.result()

        return results

    def _get_image_array(self, image: ImageInput) -> Optional[np.ndarray]:
        """
        Convert various image types to numpy array.

        Args:
            image: Input image

        Returns:
            numpy array or None if conversion fails
        """
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))[:, :, ::-1]  # RGB to BGR

        if isinstance(image, bytes):
            # Decode bytes to numpy array
            nparr = np.frombuffer(image, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if isinstance(image, str):
            # Check if base64
            if image.endswith("=") and " " not in image:
                try:
                    decoded = b64decode(bytes(image, encoding="utf8"))
                    nparr = np.frombuffer(decoded, np.uint8)
                    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    pass

            # Try as file path
            image = Path(image)

        if isinstance(image, Path):
            if image.exists():
                ext = image.suffix.lower()
                if ext == ".pdf":
                    from ...reader import PDFReader

                    with PDFReader(str(image)) as reader:
                        if reader.page_count > 0:
                            return reader[0].to_image()
                elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
                    return cv2.imread(str(image))

        return None


class AWSTextractOCRAsync(AWSTextractOCRBase):
    """
    Asynchronous AWS Textract OCR implementation.

    This class provides asynchronous methods for OCR processing using
    AWS Textract API's async client (aioboto3).

    Example:
        async_ocr = AWSTextractOCRAsync(jpeg_quality=85)

        # Process an image asynchronously
        df = await async_ocr.process_image_async("image.jpg")

        # Get raw response asynchronously
        response = await async_ocr.read_raw_async(image_array)
    """

    async def read_raw_async(self, image: ImageInput) -> TextractResponse:
        """
        Get raw AWS Textract OCR response asynchronously.

        Args:
            image: Input image in any supported format

        Returns:
            Dict response from AWS Textract API
        """
        logger.info("Using AWS Textract OCR provider (async)")

        # Construct document bytes in executor to avoid blocking
        document_bytes = await to_thread(self._construct_textract_document, image)

        # Use aioboto3 async client
        session = aioboto3.Session()
        async with session.client("textract") as client:
            response = await client.detect_document_text(Document={"Bytes": document_bytes})

        return response

    def read_raw(self, image: ImageInput) -> TextractResponse:
        """
        Synchronous wrapper for read_raw_async.

        For true async usage, use read_raw_async directly.
        """
        return asyncio.run(self.read_raw_async(image))

    def process_page(self, page: "PDFPage", page_index: int = 0) -> pd.DataFrame:
        """
        Synchronous wrapper for process_page_async.

        For true async usage, use process_page_async directly.
        """
        return asyncio.run(self.process_page_async(page, page_index))

    async def process_image_async(
        self,
        image: ImageInput,
        page_index: int = 0,
    ) -> pd.DataFrame:
        """
        Process an image asynchronously and extract OCR data.

        Args:
            image: Input image in any supported format
            page_index: Page index for the 'page' column (default: 0)

        Returns:
            pd.DataFrame with OCR data
        """
        logger.info("Processing image with AWS Textract OCR (async)")

        # Get image array for dimensions
        image_array = await to_thread(self._get_image_array, image)
        if image_array is None:
            raise ValueError("Could not get image dimensions")

        height, width = image_array.shape[:2]

        # Get raw response asynchronously
        response = await self.read_raw_async(image)

        # Parse response to DataFrame (CPU-bound, run in executor)
        df = await to_thread(self._parse_response_to_df, response, width, height, page_index)

        if df.empty:
            del image_array
            return df

        # Apply orientation correction if enabled
        if self.fix_orientation:
            df, _, _ = await to_thread(self._apply_orientation_correction, image_array, df)

        # Apply wordify
        df = await to_thread(self._wordify, df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = await to_thread(sort_df, df)

        # Free memory
        del image_array

        return df[OCR_COLUMNS] if not df.empty else df

    async def process_page_async(self, page: "PDFPage", page_index: int = 0) -> pd.DataFrame:
        """
        Process a single PDF page asynchronously.

        Args:
            page: PDFPage object to process
            page_index: Zero-based page index for the 'page' column

        Returns:
            pd.DataFrame with standardized OCR columns
        """
        logger.info(f"Processing page {page_index} with AWS Textract OCR (async)")

        # Get page image in executor (may involve PDF rendering)
        image = await to_thread(page.to_image)
        height, width = image.shape[:2]

        # Get raw response asynchronously
        response = await self.read_raw_async(image)

        # Parse response to DataFrame
        df = await to_thread(self._parse_response_to_df, response, width, height, page_index)

        if df.empty:
            logger.debug(f"No text found on page {page_index}")
            del image
            return self.create_empty_df()

        # Apply orientation correction if enabled
        if self.fix_orientation:
            df, _, _ = await to_thread(self._apply_orientation_correction, image, df)

        # Apply wordify
        df = await to_thread(self._wordify, df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = await to_thread(sort_df, df)

        # Free memory
        del image

        return df[OCR_COLUMNS] if not df.empty else df

    async def process_document_async(self, reader: "PDFReader") -> List[pd.DataFrame]:
        """
        Process all pages in a PDF document asynchronously.

        Pages are processed concurrently for better performance.

        Args:
            reader: PDFReader instance

        Returns:
            List of DataFrames, one per page
        """
        logger.info(f"Processing document with AWS Textract OCR async ({reader.page_count} pages)")

        # Process pages concurrently
        tasks = [self.process_page_async(page, page_index=i) for i, page in enumerate(reader)]

        return await asyncio.gather(*tasks)

    def _get_image_array(self, image: ImageInput) -> Optional[np.ndarray]:
        """
        Convert various image types to numpy array.

        Args:
            image: Input image

        Returns:
            numpy array or None if conversion fails
        """
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))[:, :, ::-1]  # RGB to BGR

        if isinstance(image, bytes):
            # Decode bytes to numpy array
            nparr = np.frombuffer(image, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if isinstance(image, str):
            # Check if base64
            if image.endswith("=") and " " not in image:
                try:
                    decoded = b64decode(bytes(image, encoding="utf8"))
                    nparr = np.frombuffer(decoded, np.uint8)
                    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    pass

            # Try as file path
            image = Path(image)

        if isinstance(image, Path):
            if image.exists():
                ext = image.suffix.lower()
                if ext == ".pdf":
                    from ...reader import PDFReader

                    with PDFReader(str(image)) as reader:
                        if reader.page_count > 0:
                            return reader[0].to_image()
                elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
                    return cv2.imread(str(image))

        return None
