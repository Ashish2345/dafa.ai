"""
Standalone document parsing endpoint.

Accepts a document (upload or URL) and returns per-page parsed text
synchronously. No DB persistence, no background ingestion.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, status
from loguru import logger

from app.api.v1.forms import ParseOnlyFormData
from app.config.request_mapping import RequestConfigBuilder
from app.services.download import FileHandler
from app.services.parsers.factory import ParserFactory
from app.utils.auth import get_current_user
from app.utils.exceptions import AppException

router = APIRouter(prefix="/parse", tags=["parse"])


@router.post(
    "",
    summary="Parse a document and return its text per page",
)
async def parse_document(
    file: Optional[UploadFile] = File(None),
    form_data: ParseOnlyFormData = Depends(ParseOnlyFormData.as_form()),
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Parse a document synchronously and return per-page parsed text."""
    file_path: Optional[str] = None
    try:
        handler = FileHandler()
        file_path = await handler.process(
            file_type=form_data.file_type,
            file=file,
            file_url=form_data.file_url,
        )

        request_config = RequestConfigBuilder.from_form_data(form_data.model_dump())

        # Nepali PDFs: the embedded text layer often has broken ToUnicode CMaps,
        # so DigitalOCR produces scrambled characters. Route through Google Vision
        # on rasterized pages instead.
        if (form_data.language or "").lower() == "ne":
            request_config.pdf_config.force_ocr = True
            if not form_data.ocr_provider:
                request_config.pdf_config.ocr_provider = "google"
            if not form_data.ocr_languages:
                request_config.pdf_config.ocr_languages = ["ne"]

        factory = ParserFactory(request_config)
        result = await factory.parse(Path(file_path))

        pages = {
            f"page_{page.page_number}": page.page_content or ""
            for page in result.pages
            if page.page_number is not None
        }

        logger.info(
            f"Parse complete: {Path(file_path).name} "
            f"({len(result.pages)} pages, {len(result.content)} chars)"
        )
        return {
            "file_type": result.file_type,
            "pages": pages,
        }

    except ValueError as e:
        raise AppException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="E_INVALID_INPUT",
            message=str(e),
        ) from e
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_PARSE_FAILED",
            message=str(e),
        ) from e
    finally:
        if file_path:
            Path(file_path).unlink(missing_ok=True)
