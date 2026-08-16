"""Embeddings utilities for intelligence pipelines."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from typing import Any

import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, Any] = {}
SentenceTransformer: Any | None = None


def _fallback_embedding(text: str, dimension: int = 384) -> list[float]:
    """Deterministic lightweight embedding for offline and test environments."""
    cleaned = re.findall(r"[a-z0-9]+", text.lower())
    if not cleaned:
        return [0.0] * dimension

    vector = np.zeros(dimension, dtype=float)
    for token in cleaned:
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dimension
        vector[idx] += 1.0

    for left, right in zip(cleaned, cleaned[1:]):
        token_pair = f"{left} {right}"
        idx = int(hashlib.md5(token_pair.encode("utf-8")).hexdigest(), 16) % dimension
        vector[idx] += 0.5

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def get_embedding_model(model_name: str = _DEFAULT_MODEL, device: str = "cpu") -> Any:
    global SentenceTransformer
    if os.environ.get("MEETSYNC_USE_LOCAL_EMBEDDINGS", "1") == "1":
        return None

    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _SentenceTransformer

        SentenceTransformer = _SentenceTransformer

    cache_key = f"{model_name}@{device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[cache_key]


def embed(
    texts: Iterable[str],
    model_name: str = _DEFAULT_MODEL,
    device: str = "cpu",
    batch_size: int = 32,
) -> list[list[float]]:
    """Return normalized embedding vectors for the provided texts."""
    texts_list = list(texts)
    if not texts_list:
        return []

    if os.environ.get("MEETSYNC_USE_LOCAL_EMBEDDINGS", "1") == "1":
        return [_fallback_embedding(text) for text in texts_list]

    try:
        model = get_embedding_model(model_name=model_name, device=device)
        if model is None:
            return [_fallback_embedding(text) for text in texts_list]
        embeddings = model.encode(
            texts_list,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in embeddings]
    except Exception:  # pragma: no cover - fallback when model loading fails in offline env
        return [_fallback_embedding(text) for text in texts_list]


def rank_by_similarity(
    query: str,
    candidates: Iterable[str],
    model_name: str = _DEFAULT_MODEL,
    device: str = "cpu",
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Rank candidate texts by cosine similarity to the query.

    Returns a list of (candidate, similarity) sorted descending by similarity.
    """
    candidates_list = list(candidates)
    if not candidates_list:
        return []

    # Embed query + candidates together so model batching is efficient and
    # vectors are comparable in the same space.
    texts = [query] + candidates_list
    embs = embed(texts, model_name=model_name, device=device, batch_size=batch_size)

    q = np.asarray(embs[0], dtype=float)
    cembs = np.asarray(embs[1:], dtype=float)

    # If embeddings are normalized (embed() requests normalization), cosine
    # similarity is just the dot product. Compute robustly anyway.
    q_norm = np.linalg.norm(q)
    c_norms = np.linalg.norm(cembs, axis=1)
    # Avoid division by zero
    denom = c_norms * (q_norm or 1.0)
    sims = np.dot(cembs, q) / np.where(denom == 0, 1.0, denom)

    results: list[tuple[str, float]] = []
    for candidate, sim in zip(candidates_list, sims.tolist()):
        results.append((candidate, float(sim)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results