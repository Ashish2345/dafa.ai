"""
Main router for API v1 endpoints.
5 route groups: health, auth, documents, query, waitlist.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import auth, documents, health, query
from app.api.v1.endpoints.waitlist import router as waitlist_router

api_router = APIRouter()

# Public routes
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(waitlist_router)

# Protected routes
api_router.include_router(documents.router)
api_router.include_router(query.router)