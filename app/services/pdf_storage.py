"""
PDF storage service.

Handles saving PDFs and converting them to images, storing both in MongoDB.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from app.db.repositories.file_storage import FileStorageRepository
from app.services.pdf_converter import PDFConverter
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)


class PDFStorageService:
    """Service for storing PDFs and their page images in MongoDB."""

    def __init__(self, dpi: int = 300, quality: int = 100):
        """
        Initialize PDF storage service.

        Args:
            dpi: DPI for image conversion (default: 300)
            quality: JPEG quality (1-100, default: 100)
        """
        self.file_storage = FileStorageRepository()
        self.pdf_converter = PDFConverter(dpi=dpi, quality=quality)

    async def save_pdf_and_images(
        self,
        pdf_path: str,
        document_id: str,
        filename: Optional[str] = None,
        max_pages: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Save PDF and convert pages to images, storing both in MongoDB.

        Args:
            pdf_path: Path to the PDF file
            document_id: Document identifier
            filename: Original filename (optional)
            max_pages: Maximum number of pages to convert (None = all pages)
            metadata: Additional metadata to store (optional)

        Returns:
            Dictionary with:
            - pdf_file_id: GridFS file_id for the PDF
            - image_file_ids: List of GridFS file_ids for page images
            - page_count: Number of pages converted

        Raises:
            AppException: If storage or conversion fails
        """
        try:
            logger.info(f"Saving PDF and converting to images: document_id={document_id}")

            # Step 1: Save the PDF to GridFS
            pdf_file_id = await self.file_storage.save_pdf(
                file_path=pdf_path,
                document_id=document_id,
                filename=filename,
                metadata=metadata,
            )

            # Step 2: Convert PDF pages to images
            logger.info(f"Converting PDF pages to images: document_id={document_id}")
            image_data_list = self.pdf_converter.convert_pdf_to_images(
                pdf_path=pdf_path, max_pages=max_pages
            )

            # Step 3: Save each page image to GridFS
            image_file_ids = []
            for page_number, image_data in enumerate(image_data_list, start=1):
                image_file_id = await self.file_storage.save_image(
                    image_data=image_data,
                    document_id=document_id,
                    page_number=page_number,
                    filename=f"{document_id}_page_{page_number}.jpg",
                    metadata=metadata,
                )
                image_file_ids.append(image_file_id)

            logger.info(
                f"Successfully saved PDF and {len(image_file_ids)} images: "
                f"document_id={document_id}, pdf_file_id={pdf_file_id}"
            )

            return {
                "pdf_file_id": pdf_file_id,
                "image_file_ids": image_file_ids,
                "page_count": len(image_file_ids),
            }

        except AppException:
            raise
        except Exception as e:
            logger.error(f"Error saving PDF and images: {e}")
            raise AppException(
                message=f"Failed to save PDF and images: {str(e)}",
                error_code="E_PDF_STORAGE_FAILED",
                status_code=500,
            ) from e

    async def get_pdf(self, file_id: str) -> Tuple[bytes, dict]:
        """
        Retrieve a PDF file from GridFS.

        Args:
            file_id: GridFS file_id

        Returns:
            Tuple of (file_data as bytes, metadata dict)
        """
        return await self.file_storage.get_pdf(file_id)

    async def get_image(self, file_id: str) -> Tuple[bytes, dict]:
        """
        Retrieve an image from GridFS.

        Args:
            file_id: GridFS file_id

        Returns:
            Tuple of (image_data as bytes, metadata dict)
        """
        return await self.file_storage.get_image(file_id)

    async def get_images_by_document(self, document_id: str) -> List[dict]:
        """
        Get all images for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of image metadata dicts with file_id, page_number, and storage_path
        """
        return await self.file_storage.get_images_by_document(document_id)

    async def get_pdf_by_document(self, document_id: str) -> Optional[dict]:
        """
        Get PDF file for a document.

        Args:
            document_id: Document identifier

        Returns:
            PDF metadata dict with file_id and storage_path, or None if not found
        """
        return await self.file_storage.get_pdf_by_document(document_id)

    async def get_document_structure(self, document_id: str) -> dict:
        """
        Get complete file structure for a document.

        Args:
            document_id: Document identifier

        Returns:
            Dictionary with PDF and images information:
            {
                "document_id": str,
                "pdf": dict or None,
                "images": List[dict],
                "total_files": int
            }
        """
        pdf_info = await self.file_storage.get_pdf_by_document(document_id)
        images = await self.file_storage.get_images_by_document(document_id)

        return {
            "document_id": document_id,
            "pdf": pdf_info,
            "images": images,
            "total_files": (1 if pdf_info else 0) + len(images),
            "storage_structure": {
                "pdf_path": f"documents/{document_id}/pdf/" if pdf_info else None,
                "images_path": f"documents/{document_id}/images/" if images else None,
            },
        }

    async def delete_document_files(self, document_id: str) -> dict:
        """
        Delete all files (PDF and images) for a document.

        Args:
            document_id: Document identifier

        Returns:
            Dictionary with deletion results
        """
        return await self.file_storage.delete_document_files(document_id)
