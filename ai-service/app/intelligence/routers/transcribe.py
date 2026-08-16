from fastapi import APIRouter, HTTPException, status

from app.intelligence.schemas import TranscribeRequest, Transcript

transcribe_router = APIRouter(prefix="/transcribe", tags=["intelligence"])


@transcribe_router.get("", summary="Transcribe router health check")
def transcribe_health_check() -> dict[str, str]:
    return {"status": "transcribe router ok"}


@transcribe_router.post("", response_model=Transcript, summary="Transcribe an audio recording")
async def transcribe_audio(request: TranscribeRequest) -> Transcript:
    """Contract placeholder until the Whisper provider is configured."""
    _ = request
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Whisper transcription provider is not configured.",
    )