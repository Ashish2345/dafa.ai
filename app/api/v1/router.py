"""
Main router for API v1 endpoints.

Endpoint structure:
  endpoints/
    auth/          → login, register, refresh, me
    documents/     → upload, list, CRUD, images, highlights
    query/         → ask questions, streaming
    user/          → chats, preferences, starred, usage, plans, feedback
    health.py      → health check (public)
    waitlist.py    → waitlist signup (public)
"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, documents, query
from app.api.v1.endpoints.parse import router as parse_router
from app.api.v1.endpoints.studio import router as studio_router
from app.api.v1.endpoints.user import (
    chats_router,
    connected_accounts_router,
    export_router,
    feedback_router,
    plans_router,
    preferences_router,
    profile_photo_router,
    starred_router,
    team_router,
    usage_router,
)
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.waitlist import router as waitlist_router

api_router = APIRouter()

# Public routes
api_router.include_router(health_router)
api_router.include_router(auth.router)
api_router.include_router(waitlist_router)

# Protected routes (require JWT)
api_router.include_router(documents.router)
api_router.include_router(documents.highlights_router)
api_router.include_router(parse_router)
api_router.include_router(query.router)
api_router.include_router(query.stream_router)
api_router.include_router(studio_router)
api_router.include_router(preferences_router)
api_router.include_router(starred_router)
api_router.include_router(chats_router)
api_router.include_router(feedback_router)
api_router.include_router(plans_router)
api_router.include_router(usage_router)
api_router.include_router(team_router)
api_router.include_router(profile_photo_router)
api_router.include_router(export_router)
api_router.include_router(connected_accounts_router)
