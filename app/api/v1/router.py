"""
Main router for API v1 endpoints.

Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import ask, collections, documents, files, health, ingest, parse

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router)
api_router.include_router(parse.router)
api_router.include_router(ingest.router)
api_router.include_router(ask.router)
api_router.include_router(collections.router)
api_router.include_router(documents.router)
api_router.include_router(files.router)