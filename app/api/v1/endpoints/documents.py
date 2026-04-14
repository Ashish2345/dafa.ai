"""
Document management endpoints.

Handles upload/ingest, listing, retrieval, and file serving.
Merges old ingest.py, collections.py, files.py, and documents.py.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.api.v1.forms import ParseFormData
from app.config.request_mapping import RequestConfigBuilder
from app.db.mongodb import get_database
from app.db.repositories.document_repository import DocumentRepository
from app.db.repositories.file_storage import FileStorageRepository
from app.db.repositories.ocr_bbox_repository import OcrBboxRepository
from app.db.repositories.page_index_repository import PageIndexRepository
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.processing import DocumentProcessor, MetadataExtractor
from app.services.parsers.factory import ParserFactory
from app.services.retrieval.factory import RetrievalFactory
from app.utils.auth import get_current_user
from app.utils.exceptions import AppException


async def get_page_index_repository() -> PageIndexRepository:
    """FastAPI dependency that returns a PageIndexRepository backed by the app database."""
    db = await get_database()
    return PageIndexRepository(db)


async def _run_ingestion_background(
    file_path: str,
    document_id: str,
    filename: str,
    form_data: ParseFormData,
) -> None:
    """Background task: runs the full ingestion pipeline and cleans up the temp file."""
    try:
        db = await get_database()
        doc_repo = DocumentRepository(db)

        async def on_progress(step: str) -> None:
            await doc_repo.update_status(document_id, "processing", step=step)

        language = getattr(form_data, "language", "en")
        strategy_name = getattr(form_data, "strategy", None)
        strategy = await RetrievalFactory.get_strategy(strategy_name)

        form_dict = form_data.model_dump()
        if language == "ne" and not form_data.ocr_languages:
            form_dict["ocr_languages"] = ["ne"]
        request_config = RequestConfigBuilder.from_form_data(form_dict)

        pipeline = IngestionPipeline(
            parser_factory=ParserFactory(request_config),
            processor=DocumentProcessor(),
            metadata_extractor=MetadataExtractor(language=language),
            file_storage=FileStorageRepository(db),
            document_repo=doc_repo,
            ocr_bbox_repo=OcrBboxRepository(db),
        )

        result = await pipeline.run(
            file_path=Path(file_path),
            filename=filename,
            document_id=document_id,
            strategy=strategy,
            language=language,
            on_progress=on_progress,
        )

        if result.get("status") == "failed":
            await doc_repo.update_status(
                document_id, "failed", step="Failed", error=result.get("error", "Unknown error")
            )
        else:
            logger.info(f"Background ingestion complete: {document_id}")

    except Exception as e:
        logger.error(f"Background ingestion failed for {document_id}: {e}")
        try:
            db = await get_database()
            await DocumentRepository(db).update_status(
                document_id, "failed", step="Failed", error=str(e)
            )
        except Exception:
            pass
    finally:
        Path(file_path).unlink(missing_ok=True)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", summary="Upload and ingest a document", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    current_user: dict = Depends(get_current_user),
):
    """Upload a PDF and start ingestion in the background.

    Returns immediately with ``status: processing``.
    Poll ``GET /documents/{document_id}`` to track progress via the
    ``status`` and ``progress_step`` fields.
    """
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

        document_id = str(uuid.uuid4())
        filename = original_filename or Path(file_path).name

        # Create the record now so polling works immediately
        db = await get_database()
        await DocumentRepository(db).save_initial(
            document_id, filename, category=form_data.category, title=form_data.title,
        )

        # Hand off to background — temp file is deleted by the task when done
        background_tasks.add_task(
            _run_ingestion_background,
            file_path,
            document_id,
            filename,
            form_data,
        )

        logger.info(f"Ingestion queued: {document_id} ({filename})")
        return {
            "document_id": document_id,
            "filename": filename,
            "status": "processing",
            "message": "Ingestion started. Poll GET /documents/{document_id} for progress.",
        }

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
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: dict = Depends(get_current_user),
):
    """List all ingested documents, optionally filtered by category."""
    db = await get_database()
    repo = DocumentRepository(db)
    documents = await repo.list_all(skip=skip, limit=limit, category=category)
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
    """Serve the original PDF from GridFS. Enforces per-plan daily download quota."""
    from fastapi import HTTPException
    from app.config.plan_loader import plan_catalog
    from app.db.repositories.usage_repository import UsageRepository
    from app.db.repositories.user_repository import UserRepository

    db = await get_database()
    user_id = current_user["sub"]

    # ─── Plan quota enforcement ──────────────────────────────────────
    user_doc = await UserRepository(db).get_by_id(user_id)
    plan_id = (user_doc or {}).get("plan") or plan_catalog.default_plan_id
    plan = plan_catalog.get(plan_id)

    usage_repo = UsageRepository(db)
    allowed, used, _ = await usage_repo.check_quota(
        user_id, "pdf_download", plan.limits.pdf_downloads_per_day
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "plan_limit_reached",
                "action": "pdf_download",
                "plan_id": plan_id,
                "plan_name": plan.name,
                "limit": plan.limits.pdf_downloads_per_day,
                "used": used,
                "message": (
                    f"You've used all {plan.limits.pdf_downloads_per_day} PDF downloads on your "
                    f"{plan.name} plan today. Upgrade to continue downloading."
                ),
            },
        )

    file_repo = FileStorageRepository(db)
    pdf_meta = await file_repo.get_pdf_by_document(document_id)
    if not pdf_meta:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="E_NOT_FOUND",
            message=f"PDF not found for document {document_id}",
        )
    pdf_bytes, _ = await file_repo.get_pdf(pdf_meta["file_id"])

    # Count this download against the user's daily quota
    await usage_repo.increment(user_id, "pdf_download")

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{pdf_meta.get("original_filename", "document.pdf")}"'},
    )


@router.get("/{document_id}/image/{page_number}", summary="Get document page as image")
async def get_document_image(
    document_id: str,
    page_number: int,
    scale: float = Query(1.5, ge=0.5, le=3.0, description="Render scale factor"),
    current_user: dict = Depends(get_current_user),
):
    """Serve a pre-rendered page image from GridFS, falling back to PDF rendering."""
    db = await get_database()
    file_repo = FileStorageRepository(db)

    # Try to serve the pre-rendered image from GridFS
    result = await file_repo.get_image_by_page(document_id, page_number)
    if result:
        img_bytes, _ = result
        total_pages = await file_repo.count_images_by_document(document_id)
        return Response(
            content=img_bytes,
            media_type="image/jpeg",
            headers={
                "X-Total-Pages": str(total_pages),
                "Cache-Control": "public, max-age=3600",
            },
        )

    # Fallback: render from PDF on-the-fly
    import fitz

    pdf_meta = await file_repo.get_pdf_by_document(document_id)
    if not pdf_meta:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="E_NOT_FOUND",
            message=f"PDF not found for document {document_id}",
        )
    pdf_bytes, _ = await file_repo.get_pdf(pdf_meta["file_id"])

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    if page_number < 1 or page_number > total_pages:
        doc.close()
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="E_PAGE_NOT_FOUND",
            message=f"Page {page_number} not found (document has {total_pages} pages)",
        )

    page = doc[page_number - 1]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg")
    doc.close()
    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={
            "X-Total-Pages": str(total_pages),
            "Cache-Control": "public, max-age=3600",
        },
    )


@router.get("/{document_id}/page-count", summary="Get total page count for a document")
async def get_document_page_count(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the total number of pages (from stored images or PDF)."""
    db = await get_database()
    file_repo = FileStorageRepository(db)

    # Fast path: count pre-stored images
    count = await file_repo.count_images_by_document(document_id)
    if count > 0:
        return {"total_pages": count}

    # Fallback: open PDF to count pages
    import fitz

    pdf_meta = await file_repo.get_pdf_by_document(document_id)
    if not pdf_meta:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="E_NOT_FOUND",
            message=f"PDF not found for document {document_id}",
        )
    pdf_bytes, _ = await file_repo.get_pdf(pdf_meta["file_id"])
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    doc.close()
    return {"total_pages": total}


@router.get("/{document_id}/highlights", summary="Get highlight bounding boxes for tree nodes")
async def get_document_highlights(
    document_id: str,
    node_ids: str = Query(..., description="Comma-separated node IDs"),
    current_user=Depends(get_current_user),
    page_index_repo: PageIndexRepository = Depends(get_page_index_repository),
):
    """Get highlight bounding boxes for specific tree nodes."""
    try:
        ids = [int(x.strip()) for x in node_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="node_ids must be comma-separated integers")

    if not ids:
        raise HTTPException(status_code=400, detail="At least one node_id required")

    result = await page_index_repo.get_node_highlights(document_id, ids)
    if result is None:
        raise HTTPException(status_code=404, detail="Document tree not found")

    return result


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
