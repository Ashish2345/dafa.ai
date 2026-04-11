"""
Main router for API v1 endpoints.
4 route groups: health, auth, documents, query.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import auth, documents, health, query

api_router = APIRouter()

# Public routes
api_router.include_router(health.router)
api_router.include_router(auth.router)

# Protected routes
api_router.include_router(documents.router)
api_router.include_router(query.router)