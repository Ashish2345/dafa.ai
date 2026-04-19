"""
Google Vision OCR implementation for non-digital (scanned) documents.

This module provides both synchronous and asynchronous Google Vision OCR
implementations with support for multiple image sources.

Example:
    Synchronous usage:
        from diu_new.ocr import GoogleVisionOCR
        from diu_new.pdf_reader import PDFReader

        ocr = GoogleVisionOCR(jpeg_quality=85, fix_orientation=True)

        # Process a PDF document
        with PDFReader("document.pdf") as reader:
            df_list = ocr.process_document(reader)

        # Or process a single image
        df = ocr.process_image("image.jpg")

    Asynchronous usage:
        from diu_new.ocr import GoogleVisionOCRAsync

        async_ocr = GoogleVisionOCRAsync()
        df = await async_ocr.process_image_async(image_array)
"""

import asyncio
import io
import os
from abc import abstractmethod
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from google.cloud import vision
from loguru import logger
from PIL import Image

from app.utils import to_thread

from ..base import OCR_COLUMNS, OCRBase
from ..utils import (
    get_orientation_angle_from_response,
    get_rotated_df,
    get_rotated_df_4point,
    remove_no_break_space,
    sort_df,
)

if TYPE_CHECKING:
    from google.cloud.vision_v1.types import AnnotateImageResponse

    from ...reader import PDFPage, PDFReader

# Type alias for supported image inputs
ImageInput = Union[bytes, str, np.ndarray, Path, Image.Image]


class GoogleVisionOCRBase(OCRBase):
    """
    Abstract base class for Google Vision OCR implementations.

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
        language_hints: Optional[List[str]] = None,
    ):
        self.jpeg_quality = jpeg_quality
        self.fix_orientation = fix_orientation
        self.reset_lines_and_sort = reset_lines_and_sort
        self.max_workers = max_workers
        # Convert language codes to Google Vision format
        # Google Vision uses ISO 639-1 codes (e.g., "ne" for Nepali, "en" for English)
        self.language_hints = self._normalize_language_codes(language_hints) if language_hints else None

    def _normalize_language_codes(self, languages: List[str]) -> List[str]:
        """
        Normalize language codes to Google Vision API format.
        
        Converts common language codes to ISO 639-1 format:
        - "nep" or "nepali" -> "ne"
        - "eng" or "english" -> "en"
        - etc.
        
        Args:
            languages: List of language codes
            
        Returns:
            Normalized language codes
        """
        if not languages:
            return []
        
        # Language code mapping
        lang_map = {
            "nep": "ne",
            "nepali": "ne",
            "eng": "en",
            "english": "en",
            "hin": "hi",
            "hindi": "hi",
        }
        
        normalized = []
        for lang in languages:
            lang_lower = lang.lower().strip()
            # Check if it's already in ISO format (2 letters)
            if len(lang_lower) == 2:
                normalized.append(lang_lower)
            elif lang_lower in lang_map:
                normalized.append(lang_map[lang_lower])
            else:
                # Try to extract 2-letter code or use as-is
                normalized.append(lang_lower[:2] if len(lang_lower) >= 2 else lang_lower)
        
        return normalized

    def _construct_vision_image(self, image: ImageInput) -> vision.Image:
        """
        Construct Google Vision Image object from various types of input.

        Supports: bytes, base64 string, URL string, numpy array,
        file path (including PDF), and PIL Image.

        Args:
            image: The input image in any supported format

        Returns:
            vision.Image object ready for API call

        Raises:
            ValueError: If image type is not supported
        """
        # Handle bytes directly
        if isinstance(image, bytes):
            return vision.Image(content=image)

        # Handle base64 string
        if isinstance(image, str):
            # Check if it's a base64 string (ends with = and no spaces)
            if image.endswith("=") and " " not in image:
                try:
                    content = b64decode(bytes(image, encoding="utf8"))
                    return vision.Image(content=content)
                except Exception:
                    pass

            # Check if it's a URL
            if image.startswith(("http://", "https://")):
                vision_image = vision.Image()
                vision_image.source.image_uri = image
                return vision_image

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
                return vision.Image(content=content)

            # Handle image files
            if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}:
                with open(image, "rb") as f:
                    content = f.read()
                return vision.Image(content=content)

            raise ValueError(f"Unsupported file format: {ext}")

        # Handle numpy array (BGR format from OpenCV)
        if isinstance(image, np.ndarray):
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, buffer = cv2.imencode(".jpg", image, encode_params)
            if not success:
                raise ValueError("Failed to encode numpy array as JPEG")
            content = buffer.tobytes()
            del buffer  # Free memory immediately
            return vision.Image(content=content)

        # Handle PIL Image
        if isinstance(image, Image.Image):
            with io.BytesIO() as buffer:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(buffer, format="JPEG", quality=self.jpeg_quality)
                content = buffer.getvalue()
            return vision.Image(content=content)

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _parse_response_to_df(self, response: "AnnotateImageResponse", page_index: int = 0) -> pd.DataFrame:
        """
        Parse Google Vision response to a DataFrame.

        Extracts word-level data with 4-point coordinates for accurate
        orientation correction.

        Args:
            response: Google Vision API response
            page_index: Page index for the 'page' column

        Returns:
            pd.DataFrame with OCR columns plus 4-point coordinates
        """
        data = []

        try:
            pages = response.full_text_annotation.pages
        except AttributeError:
            logger.warning("No text annotation found in response")
            return self.create_empty_df()

        for page in pages:
            for block_i, block in enumerate(page.blocks):
                inter_line = 0
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        # Extract word text
                        word_text = "".join([s.text for s in word.symbols])
                        word_text = remove_no_break_space(word_text)

                        if not word_text:
                            continue

                        # Get space type from last symbol
                        try:
                            detected_break = word.symbols[-1].property.detected_break
                            space_type = detected_break.type_ if detected_break else 1
                        except AttributeError:
                            space_type = 1

                        # Calculate confidence
                        confidences = [s.confidence for s in word.symbols if s.confidence]
                        confidence = np.mean(confidences) if confidences else 0.99

                        # Get bounding box vertices
                        try:
                            box = word.bounding_box
                            vertices = box.vertices
                            xs = [v.x for v in vertices]
                            ys = [v.y for v in vertices]
                        except AttributeError:
                            continue

                        # Calculate bounding box
                        x0, y0 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)

                        # Skip invalid boxes
                        if x0 >= x2 or y0 >= y2:
                            continue

                        temp_data = {
                            "Text": word_text,
                            "x0": x0,
                            "y0": y0,
                            "x2": x2,
                            "y2": y2,
                            "block": block_i,
                            "line": inter_line,
                            "space_type": space_type,
                            "page": page_index,
                            "confidence": confidence,
                            "index_sort": len(data),
                            # Store 4-point coordinates for accurate rotation
                            "point_x0": xs[0] if len(xs) > 0 else x0,
                            "point_x1": xs[1] if len(xs) > 1 else x2,
                            "point_x2": xs[2] if len(xs) > 2 else x2,
                            "point_x3": xs[3] if len(xs) > 3 else x0,
                            "point_y0": ys[0] if len(ys) > 0 else y0,
                            "point_y1": ys[1] if len(ys) > 1 else y0,
                            "point_y2": ys[2] if len(ys) > 2 else y2,
                            "point_y3": ys[3] if len(ys) > 3 else y2,
                        }

                        data.append(temp_data)

                        # Increment line on line break
                        if space_type > 1:
                            inter_line += 1

        if not data:
            return self.create_empty_df()

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def _normalize_coords_to_unit(
        df: pd.DataFrame, image_width: int, image_height: int,
    ) -> pd.DataFrame:
        """Convert pixel-space bbox coords to 0-1 normalized.

        Mirrors what ``DigitalOCR._words_to_df`` does at parse time. Without
        this, downstream layers (markdown bbox extractor, highlight endpoint,
        frontend overlay) see x/y values in image-pixel range and produce
        out-of-range bboxes that don't line up with the rendered page image.

        Args:
            df: OCR DataFrame in pixel space.
            image_width: Width in pixels of the image OCR was run against.
            image_height: Height in pixels of the same image.
        """
        if df.empty or image_width <= 0 or image_height <= 0:
            return df
        df = df.copy()
        for col in ("x0", "x2", "point_x0", "point_x1", "point_x2", "point_x3"):
            if col in df.columns:
                df[col] = df[col] / image_width
        for col in ("y0", "y2", "point_y0", "point_y1", "point_y2", "point_y3"):
            if col in df.columns:
                df[col] = df[col] / image_height
        return df

    def _apply_orientation_correction(
        self,
        image: np.ndarray,
        df: pd.DataFrame,
        response: "AnnotateImageResponse",
    ) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """
        Apply orientation correction to image and DataFrame.

        Uses the 4-point rotation method for accurate correction.
        Falls back to 2-point method if 4-point columns are missing.

        Args:
            image: Original image array
            df: DataFrame with OCR data
            response: Google Vision response (for angle detection)

        Returns:
            Tuple of (corrected_df, rotated_image, angle)
        """
        if df.empty:
            return df, image, 0.0

        angle = get_orientation_angle_from_response(response)
        logger.debug(f"Detected orientation angle: {angle}")

        if abs(angle) < 0.5:
            # No significant rotation needed
            return df, image, 0.0

        # Check if 4-point columns exist
        point_cols = [f"point_{axis}{i}" for axis in ["x", "y"] for i in range(4)]
        has_4_points = all(col in df.columns for col in point_cols)

        try:
            if has_4_points:
                logger.debug("Using 4-point rotation transformation")
                df, image_rotated = get_rotated_df_4point(image, df, angle)
            else:
                logger.debug("Using 2-point rotation transformation (fallback)")
                df, image_rotated = get_rotated_df(image, df, angle)
        except Exception as e:
            logger.warning(f"Orientation correction failed: {e}")
            return df, image, 0.0

        return df, image_rotated, angle

    def _wordify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines the rows of OCR dataframe if they are part of a single word.

        This handles cases where Google OCR splits words that should be together.

        Args:
            df: pandas.DataFrame with word-level OCR data

        Returns:
            Merged DataFrame with combined words
        """
        if df.empty:
            return df

        data = []
        text_parts: List[str] = []
        boxes: List[List[int]] = []
        confidences: List[float] = []
        weights: List[int] = []
        current_row = None

        rows = list(df.sort_index().itertuples())

        for i, row in enumerate(rows):
            text_parts.append(row.Text)
            boxes.append([row.x0, row.y0, row.x2, row.y2])
            confidences.append(row.confidence)
            weights.append(len(row.Text))
            current_row = row

            # Determine if we should break (create new word)
            should_break = False

            if row.space_type >= 1:
                should_break = True
            elif row.Text.strip() == ":":
                should_break = True

            # Check next word
            try:
                next_row = rows[i + 1]
                if next_row.Text.strip() == ":":
                    should_break = True
            except IndexError:
                should_break = True  # End of data

            if should_break and text_parts:
                # Combine current word parts
                x1s, y1s, x2s, y2s = zip(*boxes)
                combined_text = "".join(text_parts).strip()

                if combined_text:
                    weighted_conf = np.average(confidences, weights=weights) if weights else 0.99
                    data.append(
                        {
                            "index_sort": len(data),
                            "Text": combined_text,
                            "x0": min(x1s),
                            "y0": min(y1s),
                            "x2": max(x2s),
                            "y2": max(y2s),
                            "space_type": row.space_type if row.space_type > 0 else 2,
                            "line": current_row.line,
                            "block": current_row.block,
                            "page": current_row.page,
                            "confidence": weighted_conf,
                        }
                    )

                # Reset for next word
                text_parts = []
                boxes = []
                confidences = []
                weights = []

        if not data:
            return self.create_empty_df()

        return pd.DataFrame(data, columns=OCR_COLUMNS)

    @abstractmethod
    def read_raw(self, image: ImageInput) -> "AnnotateImageResponse":
        """Get raw Google Vision response. Must be implemented by subclass."""
        pass


class GoogleVisionOCR(GoogleVisionOCRBase):
    """
    Synchronous Google Vision OCR implementation.

    This class provides synchronous methods for OCR processing using
    Google Cloud Vision API.

    Example:
        ocr = GoogleVisionOCR(jpeg_quality=85)

        # Process a PDF document
        with PDFReader("document.pdf") as reader:
            df_list = ocr.process_document(reader)

        # Process a single image
        df = ocr.process_image("image.jpg")

        # Get raw response
        response = ocr.read_raw(image_array)
    """

    def read_raw(self, image: ImageInput) -> "AnnotateImageResponse":
        """
        Get raw Google Vision OCR response.

        Args:
            image: Input image in any supported format

        Returns:
            AnnotateImageResponse from Google Vision API
        """
        logger.info("Using Google OCR provider (sync)")
        vision_image = self._construct_vision_image(image)
        client = vision.ImageAnnotatorClient()
        
        # Build image context with language hints if provided
        image_context = None
        if self.language_hints:
            image_context = vision.ImageContext(language_hints=self.language_hints)
            logger.debug(f"Using language hints: {self.language_hints}")
        
        return client.document_text_detection(
            image=vision_image,
            image_context=image_context,
        )

    def process_image(
        self,
        image: ImageInput,
        page_index: int = 0,
    ) -> pd.DataFrame:
        """
        Process an image and extract OCR data.

        Supports: bytes, base64 string, URL, numpy array, file path, PIL Image.

        Args:
            image: Input image in any supported format
            page_index: Page index for the 'page' column (default: 0)

        Returns:
            pd.DataFrame with OCR data
        """
        logger.info("Processing image with Google Vision OCR")

        # Resolve image once so we have its dimensions for normalization,
        # regardless of whether orientation correction needs them.
        image_array = self._get_image_array(image)
        final_h = image_array.shape[0] if image_array is not None else 0
        final_w = image_array.shape[1] if image_array is not None else 0

        # Get raw response
        response = self.read_raw(image)

        # Parse response to DataFrame
        df = self._parse_response_to_df(response, page_index)

        if df.empty:
            if image_array is not None:
                del image_array
            return df

        # Apply orientation correction if enabled
        if self.fix_orientation and image_array is not None:
            df, rotated, _ = self._apply_orientation_correction(image_array, df, response)
            if rotated is not None:
                final_h, final_w = rotated.shape[:2]

        if image_array is not None:
            del image_array  # Free memory

        # Apply wordify to combine word parts
        df = self._wordify(df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = sort_df(df)

        # Normalize pixel coords to 0-1 (matches DigitalOCR behaviour).
        df = self._normalize_coords_to_unit(df, final_w, final_h)

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
        logger.info(f"Processing page {page_index} with Google Vision OCR")

        # Get page image (lazy loaded)
        image = page.to_image()
        final_h, final_w = image.shape[:2]

        # Get raw response
        response = self.read_raw(image)

        # Parse response to DataFrame
        df = self._parse_response_to_df(response, page_index)

        if df.empty:
            logger.debug(f"No text found on page {page_index}")
            return self.create_empty_df()

        # Apply orientation correction if enabled
        if self.fix_orientation:
            df, rotated, _ = self._apply_orientation_correction(image, df, response)
            if rotated is not None:
                final_h, final_w = rotated.shape[:2]

        # Apply wordify to combine word parts
        df = self._wordify(df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = sort_df(df)

        # Normalize pixel coords to 0-1 (matches DigitalOCR behaviour).
        df = self._normalize_coords_to_unit(df, final_w, final_h)

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
        logger.info(f"Processing document with Google Vision OCR ({reader.page_count} pages)")

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
        Convert various image types to numpy array for orientation correction.

        Args:
            image: Input image

        Returns:
            numpy array or None if conversion fails
        """
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))[:, :, ::-1]  # RGB to BGR

        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
                return cv2.imread(str(path))

        # For other types (bytes, URL, base64), we can't easily get the array
        return None


class GoogleVisionOCRAsync(GoogleVisionOCRBase):
    """
    Asynchronous Google Vision OCR implementation.

    This class provides asynchronous methods for OCR processing using
    Google Cloud Vision API's async client.

    Example:
        async_ocr = GoogleVisionOCRAsync(jpeg_quality=85)

        # Process an image asynchronously
        df = await async_ocr.process_image_async("image.jpg")

        # Get raw response asynchronously
        response = await async_ocr.read_raw_async(image_array)
    """

    async def read_raw_async(self, image: ImageInput) -> "AnnotateImageResponse":
        """
        Get raw Google Vision OCR response asynchronously.

        Args:
            image: Input image in any supported format

        Returns:
            AnnotateImageResponse from Google Vision API
        """
        logger.info("Using Google OCR provider (async)")

        # Construct vision image in executor to avoid blocking
        vision_image = await to_thread(self._construct_vision_image, image)

        # Create request
        request = vision.AnnotateImageRequest(
            image=vision_image,
            features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
        )

        # Use async client
        async with vision.ImageAnnotatorAsyncClient() as client:
            response = await client.batch_annotate_images(requests=[request])

            if response.responses:
                return response.responses[0]
            else:
                raise ValueError("No response received from Google Vision API")

    def read_raw(self, image: ImageInput) -> "AnnotateImageResponse":
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
        logger.info("Processing image with Google Vision OCR (async)")

        # Resolve image dimensions up front for normalization at the end.
        image_array = await to_thread(self._get_image_array, image)
        final_h = image_array.shape[0] if image_array is not None else 0
        final_w = image_array.shape[1] if image_array is not None else 0

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
            df, rotated, _ = await to_thread(
                self._apply_orientation_correction, image_array, df, response
            )
            if rotated is not None:
                final_h, final_w = rotated.shape[:2]

        if image_array is not None:
            del image_array

        # Apply wordify
        df = await to_thread(self._wordify, df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = await to_thread(sort_df, df)

        # Normalize pixel coords to 0-1 (matches DigitalOCR behaviour).
        df = await to_thread(self._normalize_coords_to_unit, df, final_w, final_h)

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
        logger.info(f"Processing page {page_index} with Google Vision OCR (async)")

        # Get page image in executor (may involve PDF rendering)
        image = await to_thread(page.to_image)
        final_h, final_w = image.shape[:2]

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
            df, rotated, _ = await to_thread(
                self._apply_orientation_correction, image, df, response
            )
            if rotated is not None:
                final_h, final_w = rotated.shape[:2]

        # Apply wordify
        df = await to_thread(self._wordify, df)

        # Sort and reset lines if enabled
        if self.reset_lines_and_sort and not df.empty:
            df = await to_thread(sort_df, df)

        # Normalize pixel coords to 0-1 (matches DigitalOCR behaviour).
        df = await to_thread(self._normalize_coords_to_unit, df, final_w, final_h)

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
        logger.info(f"Processing document with Google Vision OCR async ({reader.page_count} pages)")

        # Process pages concurrently
        tasks = [self.process_page_async(page, page_index=i) for i, page in enumerate(reader)]

        return await asyncio.gather(*tasks)

    def _get_image_array(self, image: ImageInput) -> Optional[np.ndarray]:
        """
        Convert various image types to numpy array for orientation correction.

        Args:
            image: Input image

        Returns:
            numpy array or None if conversion fails
        """
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))[:, :, ::-1]  # RGB to BGR

        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
                return cv2.imread(str(path))

        return None
