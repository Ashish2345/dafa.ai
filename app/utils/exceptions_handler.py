from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models.schemas import ErrorResponse

from .exceptions import AppException


async def custom_http_exception_handler(request: Request, exc: AppException):
    body = ErrorResponse(
        error=exc.error_code,
        message=exc.message,
        status="fail",
        status_code=exc.status_code,
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors and return a custom response."""
    errors = exc.errors()

    error_messages = []
    for error in errors:
        loc = " -> ".join(str(i) for i in error["loc"])
        msg = error["msg"]
        err_type = error["type"]
        error_messages.append(f"Error at '{loc}': {msg} (type: {err_type})")

    # Join all error messages into a single string
    all_error_messages = "\n".join(error_messages)

    raise AppException(
        status_code=status.HTTP_400_BAD_REQUEST,
        error_code="E_REQUEST_VALIDATION",
        message=all_error_messages,
    )
