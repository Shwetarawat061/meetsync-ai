import numpy as np
import pytest

from app.intelligence import embeddings


class DummySentenceTransformer:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ):
        assert convert_to_numpy is True
        assert show_progress_bar is False

        mapping = {
            "query": np.array([1.0, 0.0, 0.0], dtype=float),
            "similar": np.array([0.8, 0.2, 0.0], dtype=float),
            "dissimilar": np.array([0.0, 1.0, 0.0], dtype=float),
        }

        output = []
        for text in texts:
            vector = mapping.get(text, np.array([float(len(text)), 0.0, 0.0], dtype=float))
            if normalize_embeddings:
                norm = np.linalg.norm(vector) or 1.0
                vector = vector / norm
            output.append(vector)

        return np.vstack(output)


@pytest.fixture(autouse=True)
def clear_model_cache():
    embeddings._MODEL_CACHE.clear()
    yield
    embeddings._MODEL_CACHE.clear()


def test_embed_returns_empty_list_for_no_texts(monkeypatch):
    monkeypatch.setattr(embeddings, "SentenceTransformer", DummySentenceTransformer)
    assert embeddings.embed([]) == []


def test_embed_returns_normalized_vectors_and_uses_cached_model(monkeypatch):
    monkeypatch.setattr(embeddings, "SentenceTransformer", DummySentenceTransformer)
    monkeypatch.setenv("MEETSYNC_USE_LOCAL_EMBEDDINGS", "0")   # <-- add this line

    first_model = embeddings.get_embedding_model("dummy", device="cpu")
    second_model = embeddings.get_embedding_model("dummy", device="cpu")
    assert first_model is second_model

    vectors = embeddings.embed(["query", "similar"], model_name="dummy", device="cpu", batch_size=2)
    assert len(vectors) == 2
    assert all(len(vec) == 3 for vec in vectors)
    assert pytest.approx(np.linalg.norm(vectors[0]), rel=1e-6) == 1.0
    assert pytest.approx(np.linalg.norm(vectors[1]), rel=1e-6) == 1.0

def test_rank_by_similarity_sorts_candidates_by_cosine_similarity(monkeypatch):
    def fake_embed(texts, model_name, device, batch_size):
        return [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.8, 0.2, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
        ]

    monkeypatch.setattr(embeddings, "embed", fake_embed)

    results = embeddings.rank_by_similarity("query", ["similar", "dissimilar"], model_name="dummy", device="cpu")
    assert results[0][0] == "similar"
    assert results[1][0] == "dissimilar"
    assert results[0][1] > results[1][1]


def test_rank_by_similarity_returns_empty_for_no_candidates():
    assert embeddings.rank_by_similarity("query", []) == []


# ============================================================================
# Edge Case Tests: Silent Failure Prevention
# ============================================================================

def test_rank_by_similarity_handles_unicode_characters(monkeypatch):
    """Test that unicode/special characters don't cause silent failures."""
    def fake_embed(texts, model_name, device, batch_size):
        # Return normalized vectors
        vectors = []
        for text in texts:
            # Deterministic but different vectors for different texts
            h = hash(text) % 1000
            v = np.array([np.cos(h/100), np.sin(h/100), h % 100 / 100.0], dtype=float)
            v = v / (np.linalg.norm(v) or 1.0)
            vectors.append(v)
        return vectors

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    
    # Test with unicode, emojis, special chars
    results = embeddings.rank_by_similarity(
        "meeting é ñ 中文",
        ["discussion 中文", "日本語 meeting", "emoji 😀 test"],
        model_name="dummy",
        device="cpu"
    )
    
    # Should return results without error
    assert len(results) == 3
    assert all(isinstance(score, float) for _, score in results)
    assert all(-1.0 <= score <= 1.0 for _, score in results)  # Cosine similarity range


def test_rank_by_similarity_handles_identical_strings(monkeypatch):
    """Test that identical candidates don't cause silent failures."""
    def fake_embed(texts, model_name, device, batch_size):
        vectors = []
        for text in texts:
            if "query" in text or text == "duplicate":
                v = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                v = np.array([0.0, 1.0, 0.0], dtype=float)
            vectors.append(v)
        return vectors

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    
    results = embeddings.rank_by_similarity(
        "query",
        ["duplicate", "duplicate", "other"],
        model_name="dummy",
        device="cpu"
    )
    
    # Should preserve order for identical similarity scores
    assert len(results) == 3
    # Both "duplicate" entries should have identical similarity
    assert results[0][1] == results[1][1]


def test_rank_by_similarity_handles_empty_and_whitespace_strings(monkeypatch):
    """Test that empty/whitespace strings don't cause division by zero."""
    def fake_embed(texts, model_name, device, batch_size):
        # Simulate embedding behavior: empty/whitespace -> zero vector (would need handling)
        vectors = []
        for text in texts:
            if not text.strip():
                # Simulate a zero vector that gets normalized
                v = np.array([0.0, 0.0, 0.0], dtype=float)
            else:
                v = np.array([1.0, 0.0, 0.0], dtype=float)
            # Safely normalize
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)
        return vectors

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    
    # Include empty and whitespace candidates
    results = embeddings.rank_by_similarity(
        "query",
        ["   ", "", "valid candidate", "\t\n"],
        model_name="dummy",
        device="cpu"
    )
    
    # Should not crash and return all results
    assert len(results) == 4
    assert all(isinstance(score, float) for _, score in results)


def test_rank_by_similarity_handles_very_long_strings(monkeypatch):
    """Test that very long strings don't cause memory/embedding issues."""
    def fake_embed(texts, model_name, device, batch_size):
        vectors = []
        for text in texts:
            # Base vector, but scaled by text length (simulate some length effect)
            scale = min(1.0, len(text) / 10000.0)
            v = np.array([scale, 1.0 - scale, 0.5], dtype=float)
            v = v / (np.linalg.norm(v) or 1.0)
            vectors.append(v)
        return vectors

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    
    long_text = "x" * 50000
    results = embeddings.rank_by_similarity(
        "query",
        [long_text, long_text + "y", "short"],
        model_name="dummy",
        device="cpu"
    )
    
    # Should handle long strings without error
    assert len(results) == 3
    assert all(isinstance(score, float) for _, score in results)


def test_rank_by_similarity_robust_division_by_zero(monkeypatch):
    """Test that division by zero in similarity computation is handled."""
    def fake_embed(texts, model_name, device, batch_size):
        # Simulate edge case: some vectors have zero norm
        vectors = []
        for i, text in enumerate(texts):
            if i % 2 == 0:
                # Query and some candidates: normal vectors
                v = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                # Some candidates: zero vector (after normalization attempt)
                v = np.array([0.0, 0.0, 0.0], dtype=float)
            vectors.append(v)
        return vectors

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    
    results = embeddings.rank_by_similarity(
        "query",
        ["cand1", "cand2", "cand3"],
        model_name="dummy",
        device="cpu"
    )
    
    # Should handle zero-norm vectors gracefully without NaN/Inf
    assert len(results) == 3
    assert all(not np.isnan(score) and not np.isinf(score) for _, score in results)


def test_embed_with_empty_strings():
    """Test that empty strings in batch don't cause issues."""
    # Using the actual embed function with dummy model
    vectors = embeddings.embed(["", "test", ""], batch_size=2)
    
    # Should return 3 vectors, all normalized
    assert len(vectors) == 3
    for vec in vectors:
        norm = np.linalg.norm(vec)
        assert pytest.approx(norm, rel=1e-5) == 1.0 or norm == 0.0  # Either 1 or 0 (empty)


def test_rank_by_similarity_returns_correct_format():
    """Test that results always have correct (text, score) tuple format."""
    results = embeddings.rank_by_similarity(
        "test query",
        ["candidate one", "candidate two", "candidate three"],
        batch_size=4
    )
    
    # Verify format
    assert isinstance(results, list)
    assert len(results) == 3
    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 2
        text, score = item
        assert isinstance(text, str)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0  # Cosine similarity should be in [-1, 1]
    
    # Verify descending order
    for i in range(len(results) - 1):
        assert results[i][1] >= results[i+1][1]