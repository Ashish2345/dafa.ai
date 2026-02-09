"""
FastAPI application entry point.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env file early, before importing settings
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        # Try current directory
        load_dotenv(override=False)
except ImportError:
    # python-dotenv not installed, rely on pydantic-settings
    pass

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.db.mongodb import mongodb
from app.settings import settings
from app.utils.exceptions import AppException
from app.utils.exceptions_handler import custom_http_exception_handler, validation_exception_handler

VERSION = "0.1.0"

logging.basicConfig(
    level=settings.log_level,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database connection lifecycle."""
    logger.info(f"Starting {settings.app_name} in {settings.environment.value} mode")
    try:
        await mongodb.connect(settings)
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    yield

    logger.info("Shutting down application")
    await mongodb.disconnect()
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="FastAPI-based document parsing service with OCR capabilities",
        version=VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_exception_handler(AppException, custom_http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": settings.app_name,
            "version": VERSION,
            "docs": "/docs",
            "health": f"{settings.api_v1_prefix}/health",
        }

    logger.info(f"FastAPI application created: {settings.app_name}")
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
