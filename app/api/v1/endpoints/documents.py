"""
Document management endpoints.

Handles upload/ingest, listing, retrieval, and file serving.
Merges old ingest.py, collections.py, files.py, and documents.py.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.api.v1.forms import ParseFormData
from app.config.request_mapping import RequestConfigBuilder
from app.db.mongodb import get_database
from app.db.repositories.document_repository import DocumentRepository
from app.db.repositories.file_storage import FileStorageRepository
from app.db.repositories.page_index_repository import PageIndexRepository
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.processing import DocumentProcessor, MetadataExtractor
from app.services.parsers.factory import ParserFactory
from app.services.retrieval.factory import RetrievalFactory
from app.utils.auth import get_current_user
from app.utils.exceptions import AppException

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", summary="Upload and ingest a document")
async def upload_document(
    file: Optional[UploadFile] = File(None),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    current_user: dict = Depends(get_current_user),
):
    """Upload a PDF, run full ingestion pipeline with the selected strategy."""
    logger.info("Starting document upload")

    original_filename = None
    if file and file.filename:
        original_filename = file.filename
    elif form_data.file_url:
        from urllib.parse import urlparse
        parsed_url = urlparse(form_data.file_url)
        if parsed_url.path:
            original_filename = Path(parsed_url.path).name

    try:
        from app.services.download import FileHandler

        handler = FileHandler()
        file_path = await handler.process(
            file_type=form_data.file_type,
            file=file,
            file_url=form_data.file_url,
        )

        try:
            document_id = str(uuid.uuid4())
            language = getattr(form_data, "language", "en")

            strategy_name = getattr(form_data, "strategy", None)
            strategy = await RetrievalFactory.get_strategy(strategy_name)

            form_dict = form_data.model_dump()
            if language == "ne" and not form_data.ocr_languages:
                form_dict["ocr_languages"] = ["ne"]
            request_config = RequestConfigBuilder.from_form_data(form_dict)

            db = await get_database()
            pipeline = IngestionPipeline(
                parser_factory=ParserFactory(request_config),
                processor=DocumentProcessor(),
                metadata_extractor=MetadataExtractor(language=language),
                file_storage=FileStorageRepository(db),
                document_repo=DocumentRepository(db),
            )

            result = await pipeline.run(
                file_path=Path(file_path),
                filename=original_filename or Path(file_path).name,
                document_id=document_id,
                strategy=strategy,
                language=language,
            )

            return result

        finally:
            Path(file_path).unlink(missing_ok=True)

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_UPLOAD_FAILED",
            message=str(e),
        ) from e


@router.get("", summary="List all documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(get_current_user),
):
    """List all ingested documents."""
    db = await get_database()
    repo = DocumentRepository(db)
    documents = await repo.list_all(skip=skip, limit=limit)
    return {"documents": documents, "count": len(documents)}


@router.get("/{document_id}", summary="Get document details")
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get document metadata and status."""
    db = await get_database()
    repo = DocumentRepository(db)
    doc = await repo.get(document_id)
    if not doc:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="E_NOT_FOUND",
            message=f"Document {document_id} not found",
        )
    return doc


@router.get("/{document_id}/pdf", summary="Download document PDF")
async def get_document_pdf(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Serve the original PDF from GridFS."""
    db = await get_database()
    file_repo = FileStorageRepository(db)
    pdf_data = await file_repo.get_pdf_by_document(document_id)
    if not pdf_data:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="E_NOT_FOUND",
            message=f"PDF not found for document {document_id}",
        )
    return StreamingResponse(
        iter([pdf_data["data"]]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{pdf_data.get("filename", "document.pdf")}"'},
    )


@router.delete("/{document_id}", summary="Delete a document")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete document and its strategy-specific data."""
    db = await get_database()
    doc_repo = DocumentRepository(db)
    pi_repo = PageIndexRepository(db)
    file_repo = FileStorageRepository(db)

    deleted = await doc_repo.delete(document_id)
    if not deleted:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="E_NOT_FOUND",
            message=f"Document {document_id} not found",
        )

    try:
        await pi_repo.trees.delete_one({"document_id": document_id})
        await pi_repo.content.delete_one({"document_id": document_id})
    except Exception as e:
        logger.warning(f"Error cleaning up PageIndex data: {e}")

    try:
        await file_repo.delete_document_files(document_id)
    except Exception as e:
        logger.warning(f"Error cleaning up GridFS files: {e}")

    return {"status": "deleted", "document_id": document_id}
