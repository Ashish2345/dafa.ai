"""
Document parsing endpoint.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.v1.forms import ParseFormData
from app.config.request_mapping import RequestConfigBuilder
from app.models.schemas import ParsedResponse
from app.services.download import FileHandler
from app.services.parsers.factory import ParserFactory
from app.utils.security import get_api_key
from app.utils.exceptions import AppException

router = APIRouter(prefix="/parse", tags=["parse"])


@router.post("", response_model=ParsedResponse, summary="Parse document")
async def parse_document(
    file: Optional[UploadFile] = File( ),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    api_key: str = Depends(get_api_key),
):
    """
    Parse a document and return structured content.

    Supports PDF, DOCX, XLSX, and image files. The file can be provided either
    as a direct upload (file_type='file') or via a URL (file_type='url').

    Args:
        file: The document file to parse (required when file_type='file')
        form_data: Parsing configuration options (see ParseFormData for all fields)
        api_key: API key for authentication

    Returns:
        ParsedResponse with parsed content from the document
    """
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
            # Create factory and parse
            factory = ParserFactory(request_config)
            result = await factory.parse(file_path)
            return result
        finally:
            # Cleanup temp file
            Path(file_path).unlink(missing_ok=True)
    except AppException:
        raise
    except Exception as e:
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_INTERNAL_SERVER_ERROR",
            message=f"An unexpected error occurred {str(e)}",
        ) from e
