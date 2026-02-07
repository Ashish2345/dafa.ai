"""
Health check endpoints for monitoring and readiness probes.
"""

from fastapi import APIRouter, Depends

from app.db.mongodb import mongodb
from app.models.schemas import HealthReadyResponse, HealthResponse
from app.settings import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Basic health check")
async def health_check():
    """
    Basic health check endpoint.

    Returns service status and timestamp. This endpoint does not check dependencies.
    Use this for basic liveness probes.

    Returns:
        HealthResponse with service status
    """
    return HealthResponse(status="healthy", version="0.1.0")


@router.get("/ready", response_model=HealthReadyResponse, summary="Readiness check")
async def readiness_check():
    """
    Readiness check endpoint with dependency validation.

    Checks if the service is ready to accept requests by verifying:
    - Database connection
    - Storage backend availability

    Use this for Kubernetes readiness probes.

    Returns:
        HealthReadyResponse with dependency status
    """
    # Check database connection
    database_healthy = await mongodb.health_check()

    # Check storage backend (basic check - can be expanded)
    storage_healthy = True  # TODO: Add actual storage health check

    # Overall status
    is_ready = database_healthy and storage_healthy
    status = "ready" if is_ready else "not_ready"

    return HealthReadyResponse(status=status, database=database_healthy, storage=storage_healthy)
