from fastapi import APIRouter

from app.intelligence.routers.diarize import diarize_router
from app.intelligence.routers.embed import embed_router
from app.intelligence.routers.health import health_router
from app.intelligence.routers.transcribe import transcribe_router

intelligence_router = APIRouter(prefix="/intelligence", tags=["intelligence"])

intelligence_router.include_router(health_router)
intelligence_router.include_router(transcribe_router)
intelligence_router.include_router(diarize_router)
intelligence_router.include_router(embed_router)