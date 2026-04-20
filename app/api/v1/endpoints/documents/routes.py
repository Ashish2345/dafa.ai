"""
Document management endpoints.

Handles upload/ingest, listing, retrieval, and file serving.
Merges old ingest.py, collections.py, files.py, and documents.py.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from app.api.v1.forms import ParseFormData
from app.config.request_mapping import RequestConfigBuilder
from app.db.mongodb import get_database
from app.db.repositories.document_repository import DocumentRepository
from app.db.repositories.file_storage import FileStorageRepository
from app.db.repositories.ocr_bbox_repository import OcrBboxRepository
from app.db.repositories.page_index_repository import PageIndexRepository
from app.models.schemas import PageIndexTreeUpload, ParsedContentUpload
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


async def _run_parsed_ingestion_background(
    document_id: str,
    filename: str,
    parsed_payload: Dict[str, Any],
    form_data: ParseFormData,
    custom_tree: Optional[dict] = None,
    pdf_path: Optional[str] = None,
) -> None:
    """Background task: ingest directly from a pre-parsed JSON dump.

    Skips OCR + markdown conversion. Writes the markdown, word bboxes, tree,
    and document record straight to Mongo. When ``pdf_path`` is supplied
    (the client also uploaded the original PDF alongside the parsed JSON),
    the PDF and rendered page images are persisted to GridFS so the frontend
    can still show the document viewer with citation highlights.
    """
    try:
        db = await get_database()
        doc_repo = DocumentRepository(db)

        async def on_progress(step: str) -> None:
            await doc_repo.update_status(document_id, "processing", step=step)

        parsed_section = parsed_payload.get("parsed") or {}
        raw_section = parsed_payload.get("raw") or {}

        markdown = parsed_section.get("markdown", "") or ""
        parsed_lang = (parsed_section.get("language") or "").strip().lower() or None

        tree_lang = None
        if custom_tree is not None:
            raw_lang = custom_tree.get("language")
            if isinstance(raw_lang, str):
                tree_lang = raw_lang.strip().lower() or None
        form_lang = (getattr(form_data, "language", "") or "").strip().lower() or None
        language = tree_lang or parsed_lang or form_lang or "en"

        strategy_name = getattr(form_data, "strategy", None)
        strategy = await RetrievalFactory.get_strategy(strategy_name)

        pipeline = IngestionPipeline(
            processor=DocumentProcessor(),
            metadata_extractor=MetadataExtractor(language=language),
            document_repo=doc_repo,
            ocr_bbox_repo=OcrBboxRepository(db),
            file_storage=FileStorageRepository(db),
        )

        result = await pipeline.run_from_parsed_content(
            document_id=document_id,
            filename=filename,
            markdown=markdown,
            word_bboxes=raw_section.get("pages") or [],
            strategy=strategy,
            language=language,
            on_progress=on_progress,
            custom_tree=custom_tree,
            pdf_path=pdf_path,
        )

        if result.get("status") == "failed":
            await doc_repo.update_status(
                document_id, "failed", step="Failed", error=result.get("error", "Unknown error")
            )
        else:
            logger.info(f"Parsed-content ingestion complete: {document_id}")

    except Exception as e:
        logger.error(f"Parsed-content ingestion failed for {document_id}: {e}")
        try:
            db = await get_database()
            await DocumentRepository(db).update_status(
                document_id, "failed", step="Failed", error=str(e)
            )
        except Exception:
            pass
    finally:
        if pdf_path:
            Path(pdf_path).unlink(missing_ok=True)


async def _run_ingestion_background(
    file_path: str,
    document_id: str,
    filename: str,
    form_data: ParseFormData,
    custom_tree: Optional[dict] = None,
) -> None:
    """Background task: runs the full ingestion pipeline and cleans up the temp file."""
    try:
        db = await get_database()
        doc_repo = DocumentRepository(db)

        async def on_progress(step: str) -> None:
            await doc_repo.update_status(document_id, "processing", step=step)

        # Resolve language. Precedence: uploaded tree JSON > form > default.
        # The tree JSON is hand-curated per-document, so it's a stronger signal
        # than a form field (which might be defaulted by the client).
        tree_lang = None
        if custom_tree is not None:
            raw = custom_tree.get("language")
            if isinstance(raw, str):
                tree_lang = raw.strip().lower() or None
        form_lang = (getattr(form_data, "language", "") or "").strip().lower() or None
        language = tree_lang or form_lang or "en"
        logger.info(
            f"[{document_id[:8]}] language resolved: "
            f"tree={tree_lang!r} form={form_lang!r} → used={language!r}"
        )

        strategy_name = getattr(form_data, "strategy", None)
        strategy = await RetrievalFactory.get_strategy(strategy_name)

        form_dict = form_data.model_dump()
        # Keep form_dict.language in sync so RequestConfigBuilder sees the
        # resolved value (not the raw form default).
        form_dict["language"] = language
        if language == "ne" and not form_data.ocr_languages:
            form_dict["ocr_languages"] = ["ne"]
        request_config = RequestConfigBuilder.from_form_data(form_dict)

        # Nepali PDFs: embedded text layer typically has broken ToUnicode CMaps,
        # so DigitalOCR produces scrambled characters. Force the non-digital
        # path and default to Google Vision (matches /parse behavior).
        if language == "ne":
            request_config.pdf_config.force_ocr = True
            if not form_data.ocr_provider:
                request_config.pdf_config.ocr_provider = "google"

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
            custom_tree=custom_tree,
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
    page_index_file: Optional[UploadFile] = File(
        None,
        description=(
            "Optional user-supplied page-index tree JSON. When provided, the "
            "pipeline skips LLM tree generation and uses this tree instead. "
            "Only valid with strategy='page_index'."
        ),
    ),
    parsed_file: Optional[UploadFile] = File(
        None,
        description=(
            "Optional pre-parsed content JSON (output of POST /documents/parse). "
            "When provided, the pipeline SKIPS OCR + markdown conversion entirely "
            "and persists the supplied markdown + word bboxes straight to Mongo. "
            "Combine with page_index_file to also skip LLM tree generation. "
            "Only valid with strategy='page_index'."
        ),
    ),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    current_user: dict = Depends(get_current_user),
):
    """Upload a PDF and start ingestion in the background.

    Returns immediately with ``status: processing``.
    Poll ``GET /documents/{document_id}`` to track progress via the
    ``status`` and ``progress_step`` fields.

    Optional ``page_index_file`` (JSON):
        When supplied, the pipeline skips LLM tree generation and uses the
        provided hierarchical tree. Schema is validated synchronously; a
        malformed or structurally-invalid file returns 400 before ingestion
        starts. Titles in each node are matched against the OCR'd markdown;
        unmatched nodes are recorded in ``ingest_warnings`` on the document
        record (see ``GET /documents/{document_id}``). Only valid with
        ``strategy='page_index'`` (or the default); rejected with 400 when
        combined with ``strategy='vector'``.
    """
    original_filename = None
    if file and file.filename:
        original_filename = file.filename
    elif form_data.file_url:
        from urllib.parse import urlparse
        parsed_url = urlparse(form_data.file_url)
        if parsed_url.path:
            original_filename = Path(parsed_url.path).name

    # ------------------------------------------------------------------
    # Custom page-index tree: parse and validate synchronously.
    # Fail fast with 400 before we start the slow OCR background task.
    # ------------------------------------------------------------------
    logger.info(
        f"upload: page_index_file received? "
        f"{page_index_file is not None} "
        f"(filename={getattr(page_index_file, 'filename', None)!r})"
    )
    custom_tree: Optional[dict] = None
    if page_index_file is not None and getattr(page_index_file, "filename", ""):
        # Strategy compatibility — only page_index supports custom trees.
        strategy_name = (getattr(form_data, "strategy", None) or "page_index").lower()
        if strategy_name == "vector":
            raise HTTPException(
                status_code=400,
                detail="page_index_file is only valid with strategy='page_index'.",
            )

        raw = await page_index_file.read()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"page_index_file is not valid JSON: {exc.msg} "
                    f"at line {exc.lineno} col {exc.colno}"
                ),
            ) from exc

        try:
            tree_model = PageIndexTreeUpload.model_validate(parsed)
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "page_index_file failed schema validation",
                    "errors": exc.errors(),
                },
            ) from exc

        # Dump back to a plain dict (preserves extra fields like ``keywords``
        # because PageIndexTreeUpload/PageIndexNodeUpload use extra='allow').
        custom_tree = tree_model.model_dump()

    # ------------------------------------------------------------------
    # Pre-parsed content: skip OCR + markdown conversion when supplied.
    # ------------------------------------------------------------------
    parsed_payload: Optional[Dict[str, Any]] = None
    if parsed_file is not None and getattr(parsed_file, "filename", ""):
        strategy_name = (getattr(form_data, "strategy", None) or "page_index").lower()
        if strategy_name == "vector":
            raise HTTPException(
                status_code=400,
                detail="parsed_file is only valid with strategy='page_index'.",
            )

        raw = await parsed_file.read()
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"parsed_file is not valid JSON: {exc.msg} "
                    f"at line {exc.lineno} col {exc.colno}"
                ),
            ) from exc

        try:
            model = ParsedContentUpload.model_validate(decoded)
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "parsed_file failed schema validation",
                    "errors": exc.errors(),
                },
            ) from exc
        parsed_payload = model.model_dump()

    try:
        if parsed_payload is not None:
            # Pre-parsed ingestion path — skip OCR/markdown, but still save the
            # original PDF (+ rendered page images) when the caller supplies one.
            # Without the PDF the frontend won't have an image to draw citation
            # highlights on.
            pdf_path: Optional[str] = None
            if (file and file.filename) or form_data.file_url:
                from app.services.download import FileHandler

                handler = FileHandler()
                pdf_path = await handler.process(
                    file_type=form_data.file_type,
                    file=file,
                    file_url=form_data.file_url,
                )

            document_id = str(uuid.uuid4())
            filename = (
                original_filename
                or (Path(pdf_path).name if pdf_path else None)
                or getattr(parsed_file, "filename", None)
                or f"{document_id}.json"
            )

            db = await get_database()
            # Phase 14: uploads default to private + owned by the caller unless
            # an admin explicitly uploads into a public category from the CLI.
            is_admin = (current_user.get("role") or "user") == "admin"
            upload_scope = "public" if (is_admin and form_data.category not in (None, "", "workspace")) else "private"
            upload_user_id = None if upload_scope == "public" else current_user["sub"]
            await DocumentRepository(db).save_initial(
                document_id,
                filename,
                category=form_data.category,
                title=form_data.title,
                user_id=upload_user_id,
                scope=upload_scope,
            )

            background_tasks.add_task(
                _run_parsed_ingestion_background,
                document_id,
                filename,
                parsed_payload,
                form_data,
                custom_tree,
                pdf_path,
            )

            logger.info(
                f"Parsed-content ingestion queued: {document_id} ({filename}, "
                f"pages={len((parsed_payload.get('raw') or {}).get('pages') or [])}, "
                f"pdf={'yes' if pdf_path else 'no'})"
            )
            return {
                "document_id": document_id,
                "filename": filename,
                "status": "processing",
                "message": "Ingestion started. Poll GET /documents/{document_id} for progress.",
            }

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
        # Phase 14: same visibility rule as the parsed-content path above.
        is_admin = (current_user.get("role") or "user") == "admin"
        upload_scope = "public" if (is_admin and form_data.category not in (None, "", "workspace")) else "private"
        upload_user_id = None if upload_scope == "public" else current_user["sub"]
        await DocumentRepository(db).save_initial(
            document_id,
            filename,
            category=form_data.category,
            title=form_data.title,
            user_id=upload_user_id,
            scope=upload_scope,
        )

        # Hand off to background — temp file is deleted by the task when done
        background_tasks.add_task(
            _run_ingestion_background,
            file_path,
            document_id,
            filename,
            form_data,
            custom_tree,
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


@router.post("/parse", summary="Parse a document and write parsed + raw JSON to disk (no DB save)")
async def parse_document(
    file: Optional[UploadFile] = File(None),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    current_user: dict = Depends(get_current_user),
):
    """Dry-run parse: run OCR + markdown conversion and dump both layers to a
    local JSON file under ``docs/parsed/``. Returns only the written path —
    the full content is too large to ship over HTTP comfortably.

    Output file shape mirrors the DB collections exactly:

    * ``parsed.markdown``  — matches ``page_index_content.markdown``
    * ``parsed.language``  — matches ``page_index_trees.language``
    * ``raw.pages``        — list of ``{page, words}`` objects, each matching
                             one ``page_ocr_bboxes`` document (minus the
                             auto-generated ``document_id``/timestamps).
    """
    from datetime import datetime
    import re as _re

    from app.services.download import FileHandler
    from app.services.ingestion.service import IngestionService
    from app.services.processing.markdown import DocumentProcessor as MdProcessor

    handler = FileHandler()
    file_path = await handler.process(
        file_type=form_data.file_type,
        file=file,
        file_url=form_data.file_url,
    )

    try:
        language = (getattr(form_data, "language", "") or "en").strip().lower() or "en"
        form_dict = form_data.model_dump()
        form_dict["language"] = language
        if language == "ne" and not form_data.ocr_languages:
            form_dict["ocr_languages"] = ["ne"]
        request_config = RequestConfigBuilder.from_form_data(form_dict)

        parser_factory = ParserFactory(request_config)
        parsed = await parser_factory.parse(str(file_path))

        ingestion_service = IngestionService()
        original_stem = (
            Path(file.filename).stem if file and file.filename
            else (Path(file_path).stem if file_path else "uploaded")
        )
        gathered = ingestion_service.gather_document_data(
            parsed_response=parsed, document_name=original_stem,
        )
        if gathered.get("status") != "success":
            raise AppException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="E_PARSE_FAILED",
                message=gathered.get("error", "Failed to gather parsed data"),
            )

        markdown_result = MdProcessor().process_to_markdown(
            raw_ocr=gathered.get("raw_ocr", []),
            page_scalars=gathered.get("page_scalars", []),
            page_images=gathered.get("page_images", []),
        )

        # Defensive cleanup — strip any line-number artifacts (``-> N<-``) the
        # OCR processor may have left behind in the markdown. The per-page
        # strip in DocumentProcessor is the primary barrier; this is a safety
        # net so the dumped JSON is always clean regardless of upstream bugs.
        from app.services.ingestion.pipeline import _strip_line_markers

        raw_markdown = markdown_result.get("markdown", "") or ""
        cleaned_markdown = _strip_line_markers(raw_markdown)

        payload = {
            "parsed": {
                "markdown": cleaned_markdown,
                "language": language,
            },
            "raw": {
                "pages": markdown_result.get("word_bboxes", []) or [],
                "page_count": len(markdown_result.get("word_bboxes", []) or []),
            },
        }

        # Write to ./docs/parsed/<stem>_<timestamp>.json. The filesystem is the
        # app's process CWD (backend root), so this sits alongside the curated
        # fixture JSONs in ``docs/`` and is easy to inspect / diff locally.
        safe_stem = _re.sub(r"[^\w\-.]+", "_", original_stem)[:80] or "parsed"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("docs") / "parsed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_stem}_{stamp}.json"
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"Parse-only result written: {out_path} "
            f"(markdown={len(payload['parsed']['markdown'])} chars, "
            f"pages={payload['raw']['page_count']})"
        )

        return {
            "saved_to": str(out_path),
            "markdown_chars": len(payload["parsed"]["markdown"]),
            "page_count": payload["raw"]["page_count"],
            "language": language,
        }

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Parse-only request failed: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_PARSE_FAILED",
            message=str(e),
        ) from e
    finally:
        Path(file_path).unlink(missing_ok=True)


@router.get("", summary="List all documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: dict = Depends(get_current_user),
):
    """List ingested documents visible to the caller.

    Phase 14 visibility rules:
    - Public docs (``scope='public'`` or legacy records with no ``scope``) are
      visible to everyone.
    - Private docs (``scope='private'``) are visible *only* to their owner.
    - ``category='workspace'`` is a UX shorthand for "my private uploads" —
      it's translated to ``scope='private', user_id=<me>``.
    """
    db = await get_database()
    repo = DocumentRepository(db)

    user_id = current_user["sub"]

    # `workspace` category is a UX alias for the caller's private uploads.
    if category == "workspace":
        documents = await repo.list_all(
            skip=skip, limit=limit, scope="private", user_id=user_id,
        )
    else:
        # Otherwise show public docs + any private docs owned by the caller.
        documents = await repo.list_all(
            skip=skip, limit=limit, category=category, user_id=user_id,
        )
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


class _DefaultQuestionsBody(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=5, description="1-5 starter questions for this act")


@router.patch("/{document_id}/default-questions", summary="Set default starter questions for an act")
async def set_default_questions(
    document_id: str,
    body: _DefaultQuestionsBody,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Set per-act starter questions shown in the empty chat state."""
    repo = DocumentRepository(db)
    doc = await repo.get(document_id)
    if not doc:
        raise AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="E_NOT_FOUND",
            message=f"Document {document_id} not found",
        )
    await repo.collection.update_one(
        {"document_id": document_id},
        {"$set": {"default_questions": body.questions}},
    )
    return {"status": "ok", "default_questions": body.questions}


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

    original_name = pdf_meta.get("original_filename", "document.pdf")
    # RFC 5987: use filename* for non-ASCII names, ASCII fallback for filename
    from urllib.parse import quote
    ascii_name = original_name.encode("ascii", "ignore").decode("ascii") or "document.pdf"
    utf8_name = quote(original_name, safe="")
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"},
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


# Old section-level highlights endpoint removed — replaced by
# app/api/v1/endpoints/highlights.py which returns per-line word-level bboxes.


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
