"""
File download endpoints.

Provides endpoints for downloading PDFs and page images stored in MongoDB GridFS.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import Response

from app.services.pdf_storage import PDFStorageService
from app.utils.exceptions import AppException
from app.utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/pdf/{file_id}", summary="Download PDF file")
async def download_pdf(
    file_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Download a PDF file from MongoDB GridFS.

    Args:
        file_id: GridFS file_id of the PDF
        api_key: API key for authentication

    Returns:
        PDF file as binary response with appropriate headers
    """
    try:
        storage_service = PDFStorageService()
        pdf_data, metadata = await storage_service.get_pdf(file_id)

        filename = metadata.get("original_filename", f"document_{file_id}.pdf")

        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_data)),
            },
        )

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download PDF: {str(e)}",
        )


@router.get("/image/{file_id}", summary="Download image file")
async def download_image(
    file_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Download an image file from MongoDB GridFS.

    Args:
        file_id: GridFS file_id of the image
        api_key: API key for authentication

    Returns:
        Image file as binary response with appropriate headers
    """
    try:
        storage_service = PDFStorageService()
        image_data, metadata = await storage_service.get_image(file_id)

        filename = metadata.get("original_filename", f"image_{file_id}.jpg")

        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(image_data)),
            },
        )

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download image: {str(e)}",
        )


@router.get("/document/{document_id}/images", summary="Get all images for a document")
async def get_document_images(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get metadata for all images associated with a document.

    Args:
        document_id: Document identifier
        api_key: API key for authentication

    Returns:
        List of image metadata with file_ids and page numbers
    """
    try:
        storage_service = PDFStorageService()
        images = await storage_service.get_images_by_document(document_id)

        return {
            "document_id": document_id,
            "image_count": len(images),
            "images": images,
        }

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error getting document images: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document images: {str(e)}",
        )


@router.get("/document/{document_id}/image/{page_number}", summary="Download specific page image")
async def download_page_image(
    document_id: str,
    page_number: int = Path(..., ge=1, description="Page number (1-indexed)"),
    current_user: dict = Depends(get_current_user),
):
    """
    Download a specific page image for a document.

    Args:
        document_id: Document identifier
        page_number: Page number (1-indexed)
        api_key: API key for authentication

    Returns:
        Image file as binary response
    """
    try:
        storage_service = PDFStorageService()
        images = await storage_service.get_images_by_document(document_id)

        # Find image for the requested page
        page_image = None
        for img in images:
            if img["page_number"] == page_number:
                page_image = img
                break

        if not page_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Page {page_number} not found for document {document_id}",
            )

        # Download the image
        image_data, metadata = await storage_service.get_image(page_image["file_id"])
        filename = metadata.get("original_filename", f"{document_id}_page_{page_number}.jpg")

        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(image_data)),
            },
        )

    except HTTPException:
        raise
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error downloading page image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download page image: {str(e)}",
        )


@router.get("/document/{document_id}/pdf", summary="Download PDF by document ID")
async def download_pdf_by_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Download PDF file for a document by document_id.

    Args:
        document_id: Document identifier
        api_key: API key for authentication

    Returns:
        PDF file as binary response
    """
    try:
        storage_service = PDFStorageService()
        pdf_info = await storage_service.get_pdf_by_document(document_id)

        if not pdf_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF not found for document {document_id}",
            )

        pdf_data, metadata = await storage_service.get_pdf(pdf_info["file_id"])
        filename = metadata.get("original_filename", f"{document_id}.pdf")

        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_data)),
            },
        )

    except HTTPException:
        raise
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error downloading PDF by document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download PDF: {str(e)}",
        )


@router.get("/document/{document_id}/structure", summary="Get document file structure")
async def get_document_structure(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get complete file structure for a document including PDF and images.

    Args:
        document_id: Document identifier
        api_key: API key for authentication

    Returns:
        Dictionary with document file structure:
        {
            "document_id": str,
            "pdf": dict or None,
            "images": List[dict],
            "total_files": int,
            "storage_structure": {
                "pdf_path": str or None,
                "images_path": str or None
            }
        }
    """
    try:
        storage_service = PDFStorageService()
        structure = await storage_service.get_document_structure(document_id)
        return structure

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error getting document structure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document structure: {str(e)}",
        )
