"""Transcript-embedding endpoint contract."""

from fastapi import APIRouter

from app.intelligence.embeddings import embed
from app.intelligence.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    SegmentEmbedding,
)

embed_router = APIRouter(prefix="/embed", tags=["intelligence"])


@embed_router.post("", response_model=EmbeddingResponse, summary="Embed transcript text")
async def embed_transcript(request: EmbeddingRequest) -> EmbeddingResponse:
    vectors = embed([text.text for text in request.texts])
    embeddings = [
        SegmentEmbedding(segment_id=text.segment_id, vector=vector)
        for text, vector in zip(request.texts, vectors)
    ]
    return EmbeddingResponse(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        embeddings=embeddings,
    )