"""
Document ingestion endpoint.

This endpoint handles the offline/background process of ingesting documents
into the RAG system. It processes OCR output, converts to Markdown, chunks
the document, and prepares it for vector storage.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.v1.forms import ParseFormData
from app.config.request_mapping import RequestConfigBuilder
from app.services.download import FileHandler
from app.services.ingestion import IngestionService
from app.services.parsers.factory import ParserFactory
from app.services.processing import ProcessingService
from app.utils.exceptions import AppException
from app.utils.security import get_api_key
from loguru import logger

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", summary="Ingest document for RAG")
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    api_key: str = Depends(get_api_key),
):
    """
    Ingest and process a document for the RAG system.

    This endpoint follows a clear separation of concerns:
    1. Parses the document (OCR extraction) - ParserService
    2. Gathers/collects data - IngestionService (data collection only)
    3. Processes the document - ProcessingService (all transformations):
       - Converts OCR to Markdown
       - Chunks the document
       - Extracts metadata (Act Name, Section, Year)
    4. (Future) Generates embeddings and stores in vector DB

    This is an offline/background process that prepares documents for RAG queries.

    Args:
        file: The document file to ingest (required when file_type='file')
        form_data: Parsing configuration options
        api_key: API key for authentication

    Returns:
        Dictionary with ingestion results:
        {
            "document_id": str,
            "markdown": str,
            "chunks": List[Dict],
            "metadata": Dict,
            "status": str,
        }
    """
    logger.info("Starting document ingestion")

    # Preserve original filename before file is saved to temp
    original_filename = None
    if file and file.filename:
        original_filename = file.filename
    elif form_data.file_url:
        # Try to extract filename from URL
        from urllib.parse import urlparse
        parsed_url = urlparse(form_data.file_url)
        url_path = parsed_url.path
        if url_path:
            original_filename = Path(url_path).name

    # Get file path from upload or URL download
    try:
        handler = FileHandler()
        file_path = await handler.process(
            file_type=form_data.file_type,
            file=file,
            file_url=form_data.file_url,
        )
        form_data.file_path = file_path

        # Create RequestConfig from form data
        request_config = RequestConfigBuilder.from_form_data(form_data.model_dump())

        try:
            # Step 1: Parse the document to get OCR output
            logger.debug("Step 1: Parsing document")
            factory = ParserFactory(request_config)
            parsed_response = await factory.parse(file_path)

            # Step 2: Gather document data (ingestion - no processing)
            logger.debug("Step 2: Gathering document data")
            ingestion_service = IngestionService()
            # Use original filename if available, otherwise fall back to temp file name
            document_name = original_filename or (Path(file_path).stem if file_path else None)
            if document_name and "." in document_name:
                # Remove extension for cleaner act name extraction
                document_name = Path(document_name).stem
            
            gathered_data = ingestion_service.gather_document_data(
                parsed_response=parsed_response,
                document_name=document_name,
            )

            if gathered_data["status"] != "success":
                raise AppException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    error_code="E_INGESTION_FAILED",
                    message=gathered_data.get("error", "Failed to gather document data"),
                )

            # Step 3: Process the document (processing - all transformations)
            logger.debug("Step 3: Processing document")
            # Check if embeddings and vector storage should be enabled
            from app.settings import settings
            enable_embeddings = bool(getattr(settings, "openai_api_key", None))
            enable_vector_db = bool(getattr(settings, "qdrant_url", None))
            
            processing_service = ProcessingService(
                chunk_size=1000,
                chunk_overlap=200,
                respect_sections=True,
                generate_embeddings=enable_embeddings,
                store_in_vector_db=enable_vector_db,
            )

            processing_result = processing_service.process_document(
                raw_ocr=gathered_data["raw_ocr"],
                page_scalars=gathered_data["page_scalars"],
                page_images=gathered_data["page_images"],
                document_name=document_name,
            )

            return processing_result

        finally:
            # Cleanup temp file
            Path(file_path).unlink(missing_ok=True)

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_INTERNAL_SERVER_ERROR",
            message=f"An unexpected error occurred during ingestion: {str(e)}",
        ) from e
