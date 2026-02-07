"""
Microsoft Azure Computer Vision OCR implementation for non-digital (scanned) documents.

This module provides both synchronous and asynchronous Azure Computer Vision OCR
implementations with support for multiple image sources.

Example:
    Synchronous usage:
        from diu_new.ocr import AzureComputerVisionOCR
        from diu_new.pdf_reader import PDFReader

        ocr = AzureComputerVisionOCR(jpeg_quality=85, fix_orientation=True)

        # Process a PDF document
        with PDFReader("document.pdf") as reader:
            df_list = ocr.process_document(reader)

        # Or process a single image
        df = ocr.process_image("image.jpg")

    Asynchronous usage:
        from diu_new.ocr import AzureComputerVisionOCRAsync

        async_ocr = AzureComputerVisionOCRAsync()
        df = await async_ocr.process_image_async(image_array)
"""

import asyncio
import io
import os
import time
from abc import abstractmethod
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import aiohttp
import cv2
import numpy as np
import pandas as pd
import requests
from loguru import logger
from PIL import Image

from app.utils import to_thread

from ..base import OCR_COLUMNS, OCRBase
from ..utils import (
    break_colons,
    get_orientation_angle_from_df,
    get_rotated_df_4point,
    remove_no_break_space,
    sort_df,
)

if TYPE_CHECKING:
    from ...reader import PDFPage, PDFReader

# Type alias for supported image inputs
ImageInput = Union[bytes, str, np.ndarray, Path, Image.Image]

# Type alias for Azure response
AzureResponse = Dict[str, Any]

# Environment variable defaults
DEFAULT_ENDPOINT = os.environ.get("AZURE_COMPUTER_VISION_ENDPOINT", "")
DEFAULT_SUBSCRIPTION_KEY = os.environ.get("AZURE_SUBSCRIPTION_KEY", "")

# Polling configuration
DEFAULT_POLL_INTERVAL = 1.0  # seconds
MAX_POLL_ATTEMPTS = 120  # Maximum polling attempts (2 minutes with 1s interval)


class AzureComputerVisionOCRBase(OCRBase):
    """
    Abstract base class for Azure Computer Vision OCR implementations.

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
        endpoint: str, optional
            Azure Computer Vision endpoint URL. If None, uses
            AZURE_COMPUTER_VISION_ENDPOINT environment variable.
        subscription_key: str, optional
            Azure subscription key. If None, uses AZURE_SUBSCRIPTION_KEY
            environment variable.
        poll_interval: float
            Interval in seconds between polling attempts. Default: 1.0
    """

    def __init__(
        self,
        jpeg_quality: int = 85,
        fix_orientation: bool = True,
        reset_lines_and_sort: bool = True,
        max_workers: Optional[int] = None,
        endpoint: Optional[str] = None,
        subscription_key: Optional[str] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self.jpeg_quality = jpeg_quality
        self.fix_orientation = fix_orientation
        self.reset_lines_and_sort = reset_lines_and_sort
        self.max_workers = max_workers
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self.subscription_key = subscription_key or DEFAULT_SUBSCRIPTION_KEY
        self.poll_interval = poll_interval

        # Validate configuration
        if not self.endpoint:
            logger.warning(
                "Azure Computer Vision endpoint not configured. "
                "Set AZURE_COMPUTER_VISION_ENDPOINT environment variable or pass endpoint parameter."
            )
        if not self.subscription_key:
            logger.warning(
                "Azure subscription key not configured. "
                "Set AZURE_SUBSCRIPTION_KEY environment variable or pass subscription_key parameter."
            )

    def _get_api_url(self) -> str:
        """Get the Azure Read API URL."""
        return os.path.join(self.endpoint, "vision/v3.1/read/analyze")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Azure API requests."""
        return {
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
        }

    def _construct_azure_document(self, image: ImageInput) -> bytes:
        """
        Construct bytes for Azure Computer Vision from various types of input.

        Supports: bytes, base64 string, numpy array, file path (including PDF),
        and PIL Image.

        Args:
            image: The input image in any supported format

        Returns:
            bytes ready for Azure API call

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
        response: AzureResponse,
        page_index: int = 0,
    ) -> pd.DataFrame:
        """
        Parse Azure Computer Vision response to a DataFrame.

        Extracts word-level data with 4-point coordinates for accurate
        orientation correction.

        Azure boundingBox format: [x0, y0, x1, y1, x2, y2, x3, y3]
        Order: TL, TR, BR, BL (top-left, top-right, bottom-right, bottom-left)

        Args:
            response: Azure Computer Vision API response
            page_index: Page index for the 'page' column

        Returns:
            pd.DataFrame with OCR columns plus 4-point coordinates
        """
        data = []

        if "analyzeResult" not in response:
            logger.warning("No analyzeResult found in Azure response")
            return self.create_empty_df()

        # Get the first page's read results (single image = single page)
        read_results = response["analyzeResult"].get("readResults", [])
        if not read_results:
            logger.warning("No readResults found in Azure response")
            return self.create_empty_df()

        # Process the first read result (for single image processing)
        read_result = read_results[0]
        lines = read_result.get("lines", [])

        for line_idx, line in enumerate(lines):
            words = line.get("words", [])

            for word_idx, word in enumerate(words):
                # Extract word text
                word_text = word.get("text", "")
                word_text = remove_no_break_space(word_text)

                if not word_text:
                    continue

                # Get confidence (Azure returns 0-1)
                confidence = word.get("confidence", 1.0)

                # Get bounding box (8 values: TL, TR, BR, BL)
                bbox = word.get("boundingBox", [])
                if len(bbox) < 8:
                    continue

                # Extract 4-point coordinates
                # Order: TL(0,1), TR(2,3), BR(4,5), BL(6,7)
                point_x0, point_y0 = int(bbox[0]), int(bbox[1])  # Top-left
                point_x1, point_y1 = int(bbox[2]), int(bbox[3])  # Top-right
                point_x2, point_y2 = int(bbox[4]), int(bbox[5])  # Bottom-right
                point_x3, point_y3 = int(bbox[6]), int(bbox[7])  # Bottom-left

                # Calculate axis-aligned bounding box
                x0 = min(point_x0, point_x3)
                y0 = min(point_y0, point_y1)
                x2 = max(point_x1, point_x2)
                y2 = max(point_y2, point_y3)

                # Skip invalid boxes
                if x0 >= x2 or y0 >= y2:
                    continue

                # Determine space_type based on next word
                space_type = 1  # Default: space after word
                try:
                    next_word = words[word_idx + 1]
                    if next_word.get("text", "").startswith(":"):
                        space_type = 2  # No space before colon
                except IndexError:
                    space_type = 3  # End of line

                temp_data = {
                    "Text": word_text,
                    "x0": x0,
                    "y0": y0,
                    "x2": x2,
                    "y2": y2,
                    "block": 0,  # Azure doesn't provide block info
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
        Process DataFrame to handle colon splitting.

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

        for row in rows:
            text = row.Text
            x0, y0, x2, y2 = row.x0, row.y0, row.x2, row.y2
            space_type = row.space_type

            # Use break_colons to handle colon splitting
            for partial_text, box, partial_space in break_colons(text, [x0, y0, x2, y2], space_type):
                data.append(
                    {
                        "index_sort": len(data),
                        "Text": partial_text,
                        "x0": int(box[0]),
                        "y0": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                        "space_type": partial_space,
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

    @abstractmethod
    def read_raw(self, image: ImageInput) -> AzureResponse:
        """Get raw Azure Computer Vision response. Must be implemented by subclass."""
        pass


class AzureComputerVisionOCR(AzureComputerVisionOCRBase):
    """
    Synchronous Azure Computer Vision OCR implementation.

    This class provides synchronous methods for OCR processing using
    Azure Computer Vision Read API v3.1.

    Example:
        ocr = AzureComputerVisionOCR(jpeg_quality=85)

        # Process a PDF document
        with PDFReader("document.pdf") as reader:
            df_list = ocr.process_document(reader)

        # Process a single image
        df = ocr.process_image("image.jpg")

        # Get raw response
        response = ocr.read_raw(image_array)
    """

    def read_raw(self, image: ImageInput) -> AzureResponse:
        """
        Get raw Azure Computer Vision OCR response.

        This method submits the image to Azure and polls for results.

        Args:
            image: Input image in any supported format

        Returns:
            Dict response from Azure Computer Vision API

        Raises:
            ValueError: If the API request fails or times out
        """
        logger.info("Using Azure Computer Vision OCR provider (sync)")

        document_bytes = self._construct_azure_document(image)
        api_url = self._get_api_url()
        headers = self._get_headers()

        logger.debug(f"Submitting image to Azure Read API: {api_url}")

        # Submit the image for analysis
        response = requests.post(api_url, headers=headers, data=document_bytes, timeout=30)
        response.raise_for_status()

        # Get the operation location for polling
        operation_location = response.headers.get("Operation-Location")
        if not operation_location:
            raise ValueError("Azure API did not return Operation-Location header")

        logger.debug(f"Polling operation location: {operation_location}")

        # Poll for results
        poll_headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        analysis = {}

        for attempt in range(MAX_POLL_ATTEMPTS):
            time.sleep(self.poll_interval)

            poll_response = requests.get(operation_location, headers=poll_headers, timeout=30)
            poll_response.raise_for_status()
            analysis = poll_response.json()

            status = analysis.get("status", "")

            if status == "succeeded":
                logger.debug(f"Azure OCR completed after {attempt + 1} polling attempts")
                return analysis
            elif status == "failed":
                error_msg = analysis.get("analyzeResult", {}).get("errors", [])
                raise ValueError(f"Azure OCR failed: {error_msg}")
            elif status in ("notStarted", "running"):
                continue
            else:
                logger.warning(f"Unknown Azure OCR status: {status}")

        raise ValueError(f"Azure OCR timed out after {MAX_POLL_ATTEMPTS} polling attempts")

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
        logger.info("Processing image with Azure Computer Vision OCR")

        # Get image array for orientation correction
        image_array = self._get_image_array(image)

        # Get raw response
        response = self.read_raw(image)

        # Parse response to DataFrame
        df = self._parse_response_to_df(response, page_index)

        if df.empty:
            return df

        # Apply orientation correction if enabled
        if self.fix_orientation and image_array is not None:
            df, _, _ = self._apply_orientation_correction(image_array, df)

        # Apply wordify to handle colon splitting
        df = self._wordify(df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = sort_df(df)

        # Free memory
        if image_array is not None:
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
        logger.info(f"Processing page {page_index} with Azure Computer Vision OCR")

        # Get page image (lazy loaded)
        image = page.to_image()

        # Get raw response
        response = self.read_raw(image)

        # Parse response to DataFrame
        df = self._parse_response_to_df(response, page_index)

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
        logger.info(f"Processing document with Azure Computer Vision OCR ({reader.page_count} pages)")

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
                try:
                    results[page_idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing page {page_idx}: {e}")
                    results[page_idx] = self.create_empty_df()

        return results


class AzureComputerVisionOCRAsync(AzureComputerVisionOCRBase):
    """
    Asynchronous Azure Computer Vision OCR implementation.

    This class provides asynchronous methods for OCR processing using
    Azure Computer Vision Read API v3.1 with aiohttp.

    Example:
        async_ocr = AzureComputerVisionOCRAsync(jpeg_quality=85)

        # Process an image asynchronously
        df = await async_ocr.process_image_async("image.jpg")

        # Get raw response asynchronously
        response = await async_ocr.read_raw_async(image_array)
    """

    async def read_raw_async(self, image: ImageInput) -> AzureResponse:
        """
        Get raw Azure Computer Vision OCR response asynchronously.

        This method submits the image to Azure and polls for results using aiohttp.

        Args:
            image: Input image in any supported format

        Returns:
            Dict response from Azure Computer Vision API

        Raises:
            ValueError: If the API request fails or times out
        """
        logger.info("Using Azure Computer Vision OCR provider (async)")

        # Construct document bytes in executor to avoid blocking
        document_bytes = await to_thread(self._construct_azure_document, image)

        api_url = self._get_api_url()
        headers = self._get_headers()

        logger.debug(f"Submitting image to Azure Read API (async): {api_url}")

        async with aiohttp.ClientSession() as session:
            # Submit the image for analysis
            async with session.post(api_url, headers=headers, data=document_bytes) as response:
                response.raise_for_status()
                operation_location = response.headers.get("Operation-Location")

            if not operation_location:
                raise ValueError("Azure API did not return Operation-Location header")

            logger.debug(f"Polling operation location (async): {operation_location}")

            # Poll for results
            poll_headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
            analysis = {}

            for attempt in range(MAX_POLL_ATTEMPTS):
                await asyncio.sleep(self.poll_interval)

                async with session.get(operation_location, headers=poll_headers) as poll_response:
                    poll_response.raise_for_status()
                    analysis = await poll_response.json()

                status = analysis.get("status", "")

                if status == "succeeded":
                    logger.debug(f"Azure OCR completed after {attempt + 1} polling attempts (async)")
                    return analysis
                elif status == "failed":
                    error_msg = analysis.get("analyzeResult", {}).get("errors", [])
                    raise ValueError(f"Azure OCR failed: {error_msg}")
                elif status in ("notStarted", "running"):
                    continue
                else:
                    logger.warning(f"Unknown Azure OCR status: {status}")

        raise ValueError(f"Azure OCR timed out after {MAX_POLL_ATTEMPTS} polling attempts")

    def read_raw(self, image: ImageInput) -> AzureResponse:
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
        logger.info("Processing image with Azure Computer Vision OCR (async)")

        # Get image array for dimensions and orientation correction
        image_array = await to_thread(self._get_image_array, image)

        # Get raw response asynchronously
        response = await self.read_raw_async(image)

        # Parse response to DataFrame (CPU-bound, run in executor)
        df = await to_thread(self._parse_response_to_df, response, page_index)

        if df.empty:
            if image_array is not None:
                del image_array
            return df

        # Apply orientation correction if enabled
        if self.fix_orientation and image_array is not None:
            df, _, _ = await to_thread(self._apply_orientation_correction, image_array, df)

        # Apply wordify
        df = await to_thread(self._wordify, df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = await to_thread(sort_df, df)

        # Free memory
        if image_array is not None:
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
        logger.info(f"Processing page {page_index} with Azure Computer Vision OCR (async)")

        # Get page image in executor (may involve PDF rendering)
        image = await to_thread(page.to_image)

        # Get raw response asynchronously
        response = await self.read_raw_async(image)

        # Parse response to DataFrame
        df = await to_thread(self._parse_response_to_df, response, page_index)

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
        logger.info(f"Processing document with Azure Computer Vision OCR async ({reader.page_count} pages)")

        # Process pages concurrently
        tasks = [self.process_page_async(page, page_index=i) for i, page in enumerate(reader)]

        return await asyncio.gather(*tasks)
