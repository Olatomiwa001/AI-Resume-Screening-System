"""
Tests for embedding module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.embedding import EmbeddingManager


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_embedding_manager_initialization(temp_cache_dir):
    """Test EmbeddingManager initialization."""
    manager = EmbeddingManager(cache_dir=temp_cache_dir)
    assert manager is not None
    assert manager.model is not None


def test_encode_single_text(temp_cache_dir):
    """Test encoding a single text."""
    manager = EmbeddingManager(cache_dir=temp_cache_dir)
    texts = ["This is a test sentence."]
    embeddings = manager.encode(texts)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension


def test_encode_multiple_texts(temp_cache_dir):
    """Test encoding multiple texts."""
    manager = EmbeddingManager(cache_dir=temp_cache_dir)
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence."
    ]
    embeddings = manager.encode(texts)
    
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 384


def test_encode_empty_list(temp_cache_dir):
    """Test encoding empty list."""
    manager = EmbeddingManager(cache_dir=temp_cache_dir)
    embeddings = manager.encode([])
    assert len(embeddings) == 0


def test_embedding_caching(temp_cache_dir):
    """Test that embeddings are cached."""
    manager = EmbeddingManager(cache_dir=temp_cache_dir, enable_cache=True)
    text = "This text should be cached."
    
    # First encoding
    emb1 = manager.encode([text])
    
    # Second encoding (should use cache)
    emb2 = manager.encode([text])
    
    # Should be identical
    assert np.allclose(emb1, emb2)
    
    # Check cache file exists
    cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
    assert len(cache_files) > 0


def test_embedding_similarity(temp_cache_dir):
    """Test that similar texts have similar embeddings."""
    manager = EmbeddingManager(cache_dir=temp_cache_dir)
    
    texts = [
        "Python programming language",
        "Python coding and development",
        "The weather is nice today"
    ]
    
    embeddings = manager.encode(texts)
    
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # First two texts should be more similar to each other than to third
    assert sim_matrix[0, 1] > sim_matrix[0, 2]
    assert sim_matrix[0, 1] > sim_matrix[1, 2]