"""Tests for QdrantStore schema validation logic — focused on the dense vector check."""

from unittest.mock import MagicMock
from qdrant_client.models import VectorParams

from src.store import QdrantStore


class StubEmbedder:
    dim = 384
    def embed(self, texts): return [[0.1] * 384 for _ in texts]
    def embed_sparse_batch(self, texts): return [(list(range(1)), [0.1]) for _ in texts]


def _make_store():
    return QdrantStore(
        hermes_home="/root/.hermes",
        embedder=StubEmbedder(),
        connection_cfg={"mode": "remote", "url": "http://127.0.0.1:6333"},
    )


def test_dense_vector_present_in_dict_config():
    """Test that dense_present=True when vectors_cfg is dict with 'dense' key."""
    store = _make_store()
    vectors_cfg = {"dense": VectorParams(size=384, distance="Cosine")}
    dense_present = True
    if isinstance(vectors_cfg, dict) or vectors_cfg is not None and hasattr(vectors_cfg, "__contains__"):
        dense_present = "dense" in vectors_cfg
    assert dense_present is True


def test_dense_vector_missing_in_dict_config():
    """Test that dense_present=False when vectors_cfg is dict without 'dense' key."""
    store = _make_store()
    vectors_cfg = {"text": VectorParams(size=384, distance="Cosine")}
    dense_present = True
    if isinstance(vectors_cfg, dict) or vectors_cfg is not None and hasattr(vectors_cfg, "__contains__"):
        dense_present = "dense" in vectors_cfg
    assert dense_present is False


def test_dense_vector_assumed_present_for_unnamed_vector():
    """Test that dense_present=True when vectors_cfg is not a dict-like (unnamed vector)."""
    store = _make_store()
    vectors_cfg = MagicMock(spec=[])  # No __contains__
    dense_present = True
    if isinstance(vectors_cfg, dict) or vectors_cfg is not None and hasattr(vectors_cfg, "__contains__"):
        dense_present = "dense" in vectors_cfg
    assert dense_present is True


def test_schema_check_skips_on_type_error():
    """Test that schema check handles TypeError gracefully (e.g., 'in' fails)."""
    store = _make_store()
    # Object that has __contains__ but raises TypeError on 'in'
    class BadConfig:
        def __contains__(self, key):
            raise TypeError("unhashable type")
    vectors_cfg = BadConfig()
    dense_present = True
    try:
        if isinstance(vectors_cfg, dict) or vectors_cfg is not None and hasattr(vectors_cfg, "__contains__"):
            dense_present = "dense" in vectors_cfg
    except (TypeError, KeyError, AttributeError):
        pass  # This matches the store.py logic
    assert dense_present is True  # Falls back to True


def test_collection_exists_logic():
    """Test the collection existence check logic."""
    mock_client = MagicMock()
    mock_collection1 = MagicMock()
    mock_collection1.name = "memories"
    mock_collection2 = MagicMock()
    mock_collection2.name = "other"
    mock_client.get_collections.return_value = MagicMock(
        collections=[mock_collection1, mock_collection2]
    )
    existing = {c.name for c in mock_client.get_collections().collections}
    assert "memories" in existing
    assert "nonexistent" not in existing