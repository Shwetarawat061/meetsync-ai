"""Health-check endpoint for intelligence services."""

from fastapi import APIRouter

from app.intelligence.schemas import HealthResponse

health_router = APIRouter(tags=["intelligence"])


@health_router.get("/health", response_model=HealthResponse, summary="Check intelligence service health")
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")