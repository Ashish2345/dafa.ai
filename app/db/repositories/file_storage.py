"""
File storage repository for MongoDB GridFS.

Handles storage and retrieval of PDFs and images using MongoDB GridFS.
Organizes files in a structured folder hierarchy:
- documents/{document_id}/pdf/{filename}
- documents/{document_id}/images/page_{page_number}.jpg
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from bson import ObjectId
from pymongo.errors import PyMongoError

from app.utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class FileStorageRepository:
    """Repository for storing and retrieving files using MongoDB GridFS."""

    def __init__(self, database=None):
        """Initialize the file storage repository."""
        self.database = database
        self._fs_collection = None
        self._fs_chunks_collection = None

    def _get_pdf_path(self, document_id: str, filename: str) -> str:
        """
        Generate folder path for PDF storage.
        
        Args:
            document_id: Document identifier
            filename: Original filename
            
        Returns:
            Structured path: documents/{document_id}/pdf/{filename}
        """
        # Clean filename to remove any path separators
        clean_filename = Path(filename).name
        return f"documents/{document_id}/pdf/{clean_filename}"

    def _get_image_path(self, document_id: str, page_number: int, filename: Optional[str] = None) -> str:
        """
        Generate folder path for image storage.
        
        Args:
            document_id: Document identifier
            page_number: Page number
            filename: Optional custom filename
            
        Returns:
            Structured path: documents/{document_id}/images/page_{page_number}.jpg
        """
        if filename:
            clean_filename = Path(filename).name
            return f"documents/{document_id}/images/{clean_filename}"
        return f"documents/{document_id}/images/page_{page_number:04d}.jpg"

    async def _get_collections(self):
        """Get or create GridFS collections."""
        if self._fs_collection is None:
            if self.database is None:
                from app.db.mongodb import get_database
                self.database = await get_database()
            # GridFS uses two collections: fs.files and fs.chunks
            self._fs_collection = self.database["fs.files"]
            self._fs_chunks_collection = self.database["fs.chunks"]
        return self._fs_collection, self._fs_chunks_collection

    async def save_pdf(
        self,
        file_path: str,
        document_id: str,
        filename: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Save a PDF file to GridFS with organized folder structure.

        Args:
            file_path: Path to the PDF file
            document_id: Document identifier
            filename: Original filename (optional)
            metadata: Additional metadata to store (optional)

        Returns:
            GridFS file_id as string

        Raises:
            DatabaseError: If storage fails
        """
        try:
            fs_files, fs_chunks = await self._get_collections()
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")

            original_filename = filename or file_path_obj.name
            # Generate structured path: documents/{document_id}/pdf/{filename}
            structured_path = self._get_pdf_path(document_id, original_filename)

            # Read file content
            with open(file_path, "rb") as f:
                file_data = f.read()

            file_size = len(file_data)
            file_id = ObjectId()

            # Prepare file document
            file_doc = {
                "_id": file_id,
                "filename": structured_path,
                "length": file_size,
                "chunkSize": 255 * 1024,  # Default GridFS chunk size
                "uploadDate": datetime.utcnow(),
                "md5": None,
                "metadata": {
                    "document_id": document_id,
                    "file_type": "pdf",
                    "original_filename": original_filename,
                    "storage_path": structured_path,
                    **(metadata or {}),
                },
            }

            # Insert file document
            await fs_files.insert_one(file_doc)

            # Split file into chunks and insert
            chunk_size = 255 * 1024
            for i, chunk_start in enumerate(range(0, file_size, chunk_size)):
                chunk_data = file_data[chunk_start : chunk_start + chunk_size]
                chunk_doc = {
                    "_id": ObjectId(),
                    "files_id": file_id,
                    "n": i,
                    "data": chunk_data,
                }
                await fs_chunks.insert_one(chunk_doc)

            logger.info(
                f"Saved PDF to GridFS: document_id={document_id}, "
                f"path={structured_path}, file_id={file_id}"
            )
            return str(file_id)

        except FileNotFoundError as e:
            logger.error(f"PDF file not found: {e}")
            raise DatabaseError(
                message=f"PDF file not found: {e}",
                error_code="E_PDF_FILE_NOT_FOUND",
                status_code=404,
            ) from e
        except PyMongoError as e:
            logger.error(f"Failed to save PDF to GridFS: {e}")
            raise DatabaseError(
                message=f"Failed to save PDF to GridFS: {e}",
                error_code="E_GRIDFS_SAVE_ERROR",
                status_code=500,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error saving PDF: {e}")
            raise DatabaseError(
                message=f"Unexpected error saving PDF: {e}",
                error_code="E_PDF_SAVE_ERROR",
                status_code=500,
            ) from e

    async def save_image(
        self,
        image_data: bytes,
        document_id: str,
        page_number: int,
        filename: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Save an image to GridFS with organized folder structure.

        Args:
            image_data: Image data as bytes
            document_id: Document identifier
            page_number: Page number (0-indexed or 1-indexed)
            filename: Image filename (optional)
            metadata: Additional metadata to store (optional)

        Returns:
            GridFS file_id as string

        Raises:
            DatabaseError: If storage fails
        """
        try:
            fs_files, fs_chunks = await self._get_collections()
            original_filename = filename or f"{document_id}_page_{page_number}.jpg"
            # Generate structured path: documents/{document_id}/images/page_{page_number}.jpg
            structured_path = self._get_image_path(document_id, page_number, filename)

            file_size = len(image_data)
            file_id = ObjectId()

            # Prepare file document
            # Merge metadata but ensure file_type is always "image" for images
            merged_metadata = {**(metadata or {}), "file_type": "image"}
            file_doc = {
                "_id": file_id,
                "filename": structured_path,
                "length": file_size,
                "chunkSize": 255 * 1024,  # Default GridFS chunk size
                "uploadDate": datetime.utcnow(),
                "md5": None,
                "metadata": {
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": original_filename,
                    "storage_path": structured_path,
                    **merged_metadata,
                },
            }

            # Insert file document
            await fs_files.insert_one(file_doc)

            # Split image into chunks and insert
            chunk_size = 255 * 1024
            chunk_docs = []
            for i, chunk_start in enumerate(range(0, file_size, chunk_size)):
                chunk_data = image_data[chunk_start : chunk_start + chunk_size]
                chunk_docs.append({
                    "_id": ObjectId(),
                    "files_id": file_id,
                    "n": i,
                    "data": chunk_data,
                })
            # Insert all chunks at once for better performance
            if chunk_docs:
                await fs_chunks.insert_many(chunk_docs)

            logger.info(
                f"Saved image to GridFS: document_id={document_id}, "
                f"page={page_number}, path={structured_path}, file_id={file_id}"
            )
            return str(file_id)

        except PyMongoError as e:
            logger.error(f"Failed to save image to GridFS: {e}")
            raise DatabaseError(
                message=f"Failed to save image to GridFS: {e}",
                error_code="E_GRIDFS_IMAGE_SAVE_ERROR",
                status_code=500,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error saving image: {e}")
            raise DatabaseError(
                message=f"Unexpected error saving image: {e}",
                error_code="E_IMAGE_SAVE_ERROR",
                status_code=500,
            ) from e

    async def get_pdf(self, file_id: str) -> Tuple[bytes, dict]:
        """
        Retrieve a PDF file from GridFS.

        Args:
            file_id: GridFS file_id

        Returns:
            Tuple of (file_data as bytes, metadata dict)

        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            fs_files, fs_chunks = await self._get_collections()

            # Convert string file_id to ObjectId
            if isinstance(file_id, str):
                file_id = ObjectId(file_id)

            # Get file document
            file_info = await fs_files.find_one({"_id": file_id})
            if not file_info:
                raise DatabaseError(
                    message=f"PDF file not found: {file_id}",
                    error_code="E_PDF_NOT_FOUND",
                    status_code=404,
                )

            metadata = file_info.get("metadata", {})

            # Get all chunks and reassemble file
            chunks = await fs_chunks.find({"files_id": file_id}).sort("n", 1).to_list(length=None)
            file_data = b"".join(chunk["data"] for chunk in chunks)

            logger.info(f"Retrieved PDF from GridFS: file_id={file_id}")
            return file_data, metadata

        except PyMongoError as e:
            logger.error(f"Failed to retrieve PDF from GridFS: {e}")
            raise DatabaseError(
                message=f"Failed to retrieve PDF from GridFS: {e}",
                error_code="E_GRIDFS_RETRIEVE_ERROR",
                status_code=500,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error retrieving PDF: {e}")
            raise DatabaseError(
                message=f"Unexpected error retrieving PDF: {e}",
                error_code="E_PDF_RETRIEVE_ERROR",
                status_code=500,
            ) from e

    async def get_image(self, file_id: str) -> Tuple[bytes, dict]:
        """
        Retrieve an image from GridFS.

        Args:
            file_id: GridFS file_id

        Returns:
            Tuple of (image_data as bytes, metadata dict)

        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            fs_files, fs_chunks = await self._get_collections()

            # Convert string file_id to ObjectId
            if isinstance(file_id, str):
                file_id = ObjectId(file_id)

            # Get file document
            file_info = await fs_files.find_one({"_id": file_id})
            if not file_info:
                raise DatabaseError(
                    message=f"Image file not found: {file_id}",
                    error_code="E_IMAGE_NOT_FOUND",
                    status_code=404,
                )

            metadata = file_info.get("metadata", {})

            # Get all chunks and reassemble image
            chunks = await fs_chunks.find({"files_id": file_id}).sort("n", 1).to_list(length=None)
            image_data = b"".join(chunk["data"] for chunk in chunks)

            logger.info(f"Retrieved image from GridFS: file_id={file_id}")
            return image_data, metadata

        except PyMongoError as e:
            logger.error(f"Failed to retrieve image from GridFS: {e}")
            raise DatabaseError(
                message=f"Failed to retrieve image from GridFS: {e}",
                error_code="E_GRIDFS_IMAGE_RETRIEVE_ERROR",
                status_code=500,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error retrieving image: {e}")
            raise DatabaseError(
                message=f"Unexpected error retrieving image: {e}",
                error_code="E_IMAGE_RETRIEVE_ERROR",
                status_code=500,
            ) from e

    async def get_images_by_document(self, document_id: str) -> List[dict]:
        """
        Get all images for a document.

        Args:
            document_id: Document identifier (can be UUID document_id or document_name)

        Returns:
            List of image metadata dicts with file_id, page_number, and storage_path
        """
        try:
            fs_files, _ = await self._get_collections()

            # Find all images for this document using metadata filter
            # Query by either document_id (UUID) or document_name (for backward compatibility)
            # Images are identified by having page_number in metadata (regardless of file_type due to bug)
            cursor = fs_files.find({
                "$and": [
                    {
                        "$or": [
                            {"metadata.document_id": document_id},
                            {"metadata.document_name": document_id}
                        ]
                    },
                    {
                        "metadata.page_number": {"$exists": True, "$ne": None}
                    }
                ]
            })
            images = []
            async for file_info in cursor:
                images.append(
                    {
                        "file_id": str(file_info["_id"]),
                        "page_number": file_info.get("metadata", {}).get("page_number", 0),
                        "filename": file_info.get("filename", ""),
                        "storage_path": file_info.get("metadata", {}).get("storage_path", file_info.get("filename", "")),
                        "original_filename": file_info.get("metadata", {}).get("original_filename", file_info.get("filename", "")),
                        "metadata": file_info.get("metadata", {}),
                    }
                )

            # Sort by page number
            images.sort(key=lambda x: x["page_number"])

            logger.info(f"Found {len(images)} images for document_id={document_id}")
            return images

        except Exception as e:
            logger.error(f"Error getting images for document: {e}")
            raise DatabaseError(
                message=f"Error getting images for document: {e}",
                error_code="E_GET_IMAGES_ERROR",
                status_code=500,
            ) from e

    async def get_pdf_by_document(self, document_id: str) -> Optional[dict]:
        """
        Get PDF file for a document.

        Args:
            document_id: Document identifier (can be UUID document_id or document_name)

        Returns:
            PDF metadata dict with file_id and storage_path, or None if not found
        """
        try:
            fs_files, _ = await self._get_collections()

            # Find PDF for this document
            # Query by either document_id (UUID) or document_name (for backward compatibility)
            file_info = await fs_files.find_one({
                "$or": [
                    {"metadata.document_id": document_id},
                    {"metadata.document_name": document_id}
                ],
                "metadata.file_type": "pdf"
            })

            if not file_info:
                logger.warning(f"No PDF found for document_id={document_id}")
                return None

            return {
                "file_id": str(file_info["_id"]),
                "filename": file_info.get("filename", ""),
                "storage_path": file_info.get("metadata", {}).get("storage_path", file_info.get("filename", "")),
                "original_filename": file_info.get("metadata", {}).get("original_filename", file_info.get("filename", "")),
                "metadata": file_info.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"Error getting PDF for document: {e}")
            raise DatabaseError(
                message=f"Error getting PDF for document: {e}",
                error_code="E_GET_PDF_ERROR",
                status_code=500,
            ) from e

    async def delete_document_files(self, document_id: str) -> dict:
        """
        Delete all files (PDF and images) for a document.

        Args:
            document_id: Document identifier (can be UUID document_id or document_name)

        Returns:
            Dictionary with deletion results:
            {
                "pdf_deleted": bool,
                "images_deleted": int,
                "total_deleted": int
            }
        """
        try:
            fs_files, fs_chunks = await self._get_collections()

            deleted_count = 0

            # Delete PDF
            # Query by either document_id (UUID) or document_name (for backward compatibility)
            pdf_info = await fs_files.find_one({
                "$or": [
                    {"metadata.document_id": document_id},
                    {"metadata.document_name": document_id}
                ],
                "metadata.file_type": "pdf"
            })
            pdf_deleted = False
            if pdf_info:
                try:
                    file_id = pdf_info["_id"]
                    # Delete chunks first
                    await fs_chunks.delete_many({"files_id": file_id})
                    # Delete file document
                    await fs_files.delete_one({"_id": file_id})
                    pdf_deleted = True
                    deleted_count += 1
                    logger.info(f"Deleted PDF for document_id={document_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete PDF: {e}")

            # Delete all images
            # Query by either document_id (UUID) or document_name (for backward compatibility)
            cursor = fs_files.find({
                "$or": [
                    {"metadata.document_id": document_id},
                    {"metadata.document_name": document_id}
                ],
                "metadata.file_type": "image"
            })
            images_deleted = 0
            async for file_info in cursor:
                try:
                    file_id = file_info["_id"]
                    # Delete chunks first
                    await fs_chunks.delete_many({"files_id": file_id})
                    # Delete file document
                    await fs_files.delete_one({"_id": file_id})
                    images_deleted += 1
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete image {file_info['_id']}: {e}")

            logger.info(
                f"Deleted files for document_id={document_id}: "
                f"PDF={pdf_deleted}, Images={images_deleted}, Total={deleted_count}"
            )

            return {
                "pdf_deleted": pdf_deleted,
                "images_deleted": images_deleted,
                "total_deleted": deleted_count,
            }

        except Exception as e:
            logger.error(f"Error deleting document files: {e}")
            raise DatabaseError(
                message=f"Error deleting document files: {e}",
                error_code="E_DELETE_FILES_ERROR",
                status_code=500,
            ) from e

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from GridFS.

        Args:
            file_id: GridFS file_id

        Returns:
            True if deleted successfully
        """
        try:
            fs_files, fs_chunks = await self._get_collections()

            # Convert string file_id to ObjectId
            if isinstance(file_id, str):
                file_id = ObjectId(file_id)

            # Delete chunks first
            await fs_chunks.delete_many({"files_id": file_id})
            # Delete file document
            await fs_files.delete_one({"_id": file_id})
            logger.info(f"Deleted file from GridFS: file_id={file_id}")
            return True

        except PyMongoError as e:
            logger.error(f"Failed to delete file from GridFS: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting file: {e}")
            return False
