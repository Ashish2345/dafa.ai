"""
Main router for API v1 endpoints.
6 route groups: health, auth, documents, query, waitlist, user preferences.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import auth, chats, documents, feedback, health, highlights, plans, preferences, query, starred, usage
from app.api.v1.endpoints.waitlist import router as waitlist_router

api_router = APIRouter()

# Public routes
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(waitlist_router)

# Protected routes (require JWT)
api_router.include_router(documents.router)
api_router.include_router(highlights.router)
api_router.include_router(query.router)
api_router.include_router(preferences.router)
api_router.include_router(starred.router)
api_router.include_router(chats.router)
api_router.include_router(feedback.router)
api_router.include_router(plans.router)
api_router.include_router(usage.router)