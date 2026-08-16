"""Speaker-diarization endpoint contract."""

from fastapi import APIRouter, HTTPException, status

from app.intelligence.schemas import DiarizeRequest, Transcript

diarize_router = APIRouter(prefix="/diarize", tags=["intelligence"])


@diarize_router.post("", response_model=Transcript, summary="Assign speakers to transcript segments")
async def diarize_transcript(request: DiarizeRequest) -> Transcript:
    _ = request
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="pyannote diarization provider is not configured.",
    )