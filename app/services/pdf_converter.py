"""
PDF to image conversion service.

Converts PDF pages to images using pdftoppm (similar to process_image_api).
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from PIL import Image
import cv2
import numpy as np

from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)


class PDFConverter:
    """Service for converting PDF pages to images."""

    def __init__(self, dpi: int = 300, quality: int = 100):
        """
        Initialize PDF converter.

        Args:
            dpi: DPI for image conversion (default: 300)
            quality: JPEG quality (1-100, default: 100)
        """
        self.dpi = dpi
        self.quality = quality

    def convert_pdf_to_images(
        self, pdf_path: str, max_pages: Optional[int] = None, temp_dir: Optional[str] = None
    ) -> List[bytes]:
        """
        Convert PDF pages to images.

        Uses pdftoppm (from poppler-utils) similar to process_image_api.

        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to convert (None = all pages)
            temp_dir: Temporary directory for intermediate files (optional)

        Returns:
            List of image data as bytes (JPEG format)

        Raises:
            AppException: If conversion fails
        """
        if not Path(pdf_path).exists():
            raise AppException(
                status_code=404,
                error_code="E_FILE_NOT_FOUND",
                message=f"PDF file not found: {pdf_path}",
            )

        # Create temporary directory if not provided
        if temp_dir:
            temp_dir_obj = Path(temp_dir)
            temp_dir_obj.mkdir(parents=True, exist_ok=True)
            cleanup_temp = False
        else:
            temp_dir_obj = Path(tempfile.mkdtemp())
            cleanup_temp = True

        try:
            # Use pdftoppm to convert PDF to images
            # Similar to process_image_api/app/services/image_processing.py
            doc_id = Path(pdf_path).stem
            output_prefix = temp_dir_obj / doc_id

            # Build pdftoppm command
            cmd = [
                "pdftoppm",
                str(pdf_path),
                str(output_prefix),
                "-jpeg",
                "-r",
                str(self.dpi),
                "-q",
            ]

            # Add page limit if specified
            if max_pages:
                cmd.extend(["-l", str(max_pages)])

            logger.info(f"Converting PDF to images: {pdf_path}, DPI={self.dpi}, max_pages={max_pages}")

            # Run pdftoppm
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"pdftoppm failed, trying to repair PDF: {e}")
                # Try to repair corrupted PDF using ghostscript (similar to process_image_api)
                repaired_path = str(output_prefix.parent / f"{doc_id}_repaired.pdf")
                try:
                    subprocess.run(
                        [
                            "gs",
                            "-o",
                            repaired_path,
                            "-sDEVICE=pdfwrite",
                            "-dPDFSETTINGS=/prepress",
                            str(pdf_path),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    # Retry with repaired PDF
                    cmd[1] = repaired_path
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    pdf_path = repaired_path
                except subprocess.CalledProcessError as repair_error:
                    logger.error(f"Failed to repair PDF: {repair_error}")
                    raise AppException(
                        status_code=500,
                        error_code="E_PDF_CONVERSION_FAILED",
                        message=f"Failed to convert PDF to images: {str(e)}",
                    ) from e

            # Find all generated image files
            image_files = sorted(temp_dir_obj.glob(f"{doc_id}-*.jpg"))
            if not image_files:
                # Try alternative naming pattern
                image_files = sorted(temp_dir_obj.glob(f"{doc_id}*.jpg"))

            if not image_files:
                raise AppException(
                    status_code=500,
                    error_code="E_PDF_CONVERSION_FAILED",
                    message="No images were generated from PDF",
                )

            # Read images and convert to bytes
            images = []
            for img_file in image_files:
                try:
                    # Read image using OpenCV (similar to process_image_api)
                    img_array = cv2.imread(str(img_file))
                    if img_array is None:
                        logger.warning(f"Failed to read image: {img_file}")
                        continue

                    # Convert to JPEG bytes
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                    success, img_bytes = cv2.imencode(".jpg", img_array, encode_params)
                    if not success:
                        logger.warning(f"Failed to encode image: {img_file}")
                        continue

                    images.append(img_bytes.tobytes())
                    logger.debug(f"Converted page {len(images)}: {img_file}")

                except Exception as e:
                    logger.warning(f"Error processing image {img_file}: {e}")
                    continue

            if not images:
                raise AppException(
                    status_code=500,
                    error_code="E_PDF_CONVERSION_FAILED",
                    message="Failed to process any images from PDF",
                )

            logger.info(f"Successfully converted {len(images)} pages from PDF: {pdf_path}")
            return images

        except AppException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error converting PDF to images: {e}")
            raise AppException(
                status_code=500,
                error_code="E_PDF_CONVERSION_FAILED",
                message=f"Unexpected error converting PDF: {str(e)}",
            ) from e
        finally:
            # Cleanup temporary files
            if cleanup_temp and temp_dir_obj.exists():
                try:
                    import shutil

                    shutil.rmtree(temp_dir_obj)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory {temp_dir_obj}: {e}")

    def convert_pdf_to_images_pil(
        self, pdf_path: str, max_pages: Optional[int] = None
    ) -> List[bytes]:
        """
        Alternative method using pdf2image library (if pdftoppm is not available).

        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to convert (None = all pages)

        Returns:
            List of image data as bytes (JPEG format)
        """
        try:
            from pdf2image import convert_from_path

            logger.info(f"Converting PDF to images using pdf2image: {pdf_path}")

            # Convert PDF to PIL Images
            images_pil = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=1,
                last_page=max_pages,
                fmt="jpeg",
            )

            # Convert PIL Images to bytes
            images = []
            for img_pil in images_pil:
                # Convert to RGB if needed
                if img_pil.mode != "RGB":
                    img_pil = img_pil.convert("RGB")

                # Save to bytes
                from io import BytesIO

                img_bytes = BytesIO()
                img_pil.save(img_bytes, format="JPEG", quality=self.quality)
                images.append(img_bytes.getvalue())

            logger.info(f"Successfully converted {len(images)} pages using pdf2image")
            return images

        except ImportError:
            logger.warning("pdf2image not available, falling back to pdftoppm")
            return self.convert_pdf_to_images(pdf_path, max_pages)
        except Exception as e:
            logger.error(f"Error using pdf2image: {e}")
            # Fallback to pdftoppm method
            return self.convert_pdf_to_images(pdf_path, max_pages)
