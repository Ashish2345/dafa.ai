"""
Document ingestion endpoint.

This endpoint handles the offline/background process of ingesting documents
into the RAG system. It processes OCR output, converts to Markdown, chunks
the document, and prepares it for vector storage.
"""

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.v1.forms import ParseFormData
from app.config.request_mapping import RequestConfigBuilder
from app.services.download import FileHandler
from app.services.ingestion import IngestionService
from app.services.parsers.factory import ParserFactory
from app.services.pdf_storage import PDFStorageService
from app.services.processing import ProcessingService
from app.utils.exceptions import AppException
from app.utils.auth import get_current_user
from loguru import logger

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", summary="Ingest document for RAG")
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    current_user: dict = Depends(get_current_user),
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

        # Auto-set OCR language hints when document language is specified
        form_dict = form_data.model_dump()
        if form_data.language == "ne" and not form_data.ocr_languages:
            form_dict["ocr_languages"] = ["ne"]

        # Create RequestConfig from form data
        request_config = RequestConfigBuilder.from_form_data(form_dict)

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
            
            # Generate a document_id upfront for consistent use
            document_id = str(uuid.uuid4())
            
            gathered_data = ingestion_service.gather_document_data(
                parsed_response=parsed_response,
                document_name=document_name or document_id,
            )

            if gathered_data["status"] != "success":
                raise AppException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    error_code="E_INGESTION_FAILED",
                    message=gathered_data.get("error", "Failed to gather document data"),
                )

            # Step 3: Save PDF and convert to images (if PDF)
            pdf_storage_result = None
            if file_path.lower().endswith(".pdf"):
                logger.debug("Step 3: Saving PDF and converting to images")
                try:
                    pdf_storage_service = PDFStorageService(dpi=300, quality=100)
                    
                    pdf_storage_result = await pdf_storage_service.save_pdf_and_images(
                        pdf_path=file_path,
                        document_id=document_id,
                        filename=original_filename,
                        metadata={
                            "document_name": document_name,
                            "file_type": "pdf",
                        },
                    )
                    logger.info(
                        f"Saved PDF and {pdf_storage_result['page_count']} images: "
                        f"pdf_file_id={pdf_storage_result['pdf_file_id']}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save PDF/images to MongoDB: {e}")
                    # Continue processing even if storage fails

            # Step 4: Process the document (processing - all transformations)
            logger.debug("Step 4: Processing document")
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
                language=getattr(form_data, "language", "en"),
            )

            processing_result = processing_service.process_document(
                raw_ocr=gathered_data["raw_ocr"],
                page_scalars=gathered_data["page_scalars"],
                page_images=gathered_data["page_images"],
                document_name=document_name or document_id,
            )

            # Ensure document_id is in the result
            processing_result["document_id"] = document_id

            # Step 5: Build PageIndex tree (if enabled)
            from app.settings import settings
            if settings.use_page_index and processing_result.get("markdown"):
                logger.debug("Step 5: Building PageIndex tree")
                try:
                    from app.db.mongodb import get_database
                    from app.db.repositories.page_index_repository import PageIndexRepository
                    from app.services.page_index.service import PageIndexService

                    page_index_service = PageIndexService()
                    language = getattr(form_data, "language", "en")
                    tree = page_index_service.build_tree(
                        markdown_content=processing_result["markdown"],
                        language=language,
                    )

                    db = await get_database()
                    pi_repo = PageIndexRepository(db)
                    await pi_repo.save_tree(
                        document_id=document_id,
                        tree=tree,
                        markdown=processing_result["markdown"],
                        language=language,
                    )
                    processing_result["page_index_nodes"] = page_index_service._count_nodes(
                        tree.get("nodes", [])
                    )
                    logger.info(f"PageIndex tree built and saved: {processing_result['page_index_nodes']} nodes")
                except Exception as e:
                    logger.warning(f"PageIndex tree build failed (non-fatal): {e}")

            # Add PDF storage information to response if available
            if pdf_storage_result:
                processing_result["pdf_storage"] = {
                    "pdf_file_id": pdf_storage_result["pdf_file_id"],
                    "image_file_ids": pdf_storage_result["image_file_ids"],
                    "page_count": pdf_storage_result["page_count"],
                    "download_pdf_url": f"/api/v1/files/pdf/{pdf_storage_result['pdf_file_id']}",
                    "download_images_url": f"/api/v1/files/document/{document_id}/images",
                }

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
