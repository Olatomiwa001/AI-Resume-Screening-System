"""
Embedding Module

Manages sentence embeddings using Sentence Transformers.
Includes caching for efficiency.

Model: sentence-transformers/all-MiniLM-L6-v2
- Lightweight and fast
- 384-dimensional embeddings
- Good for semantic similarity tasks
"""

import logging
from typing import List, Optional
import numpy as np
from pathlib import Path
import joblib
import hashlib

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model ID: sentence-transformers/all-MiniLM-L6-v2
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingManager:
    """Manages text embeddings with caching."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory for caching embeddings
            enable_cache: Whether to use caching
        """
        self.model_name = model_name
        self.enable_cache = enable_cache
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("./cache/embeddings")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # In-memory cache
        self._memory_cache = {}
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Check cache
        if self.enable_cache:
            cached_embeddings = []
            texts_to_encode = []
            text_indices = []
            
            for idx, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                
                # Check memory cache
                if cache_key in self._memory_cache:
                    cached_embeddings.append((idx, self._memory_cache[cache_key]))
                else:
                    # Check disk cache
                    embedding = self._load_from_disk_cache(cache_key)
                    if embedding is not None:
                        cached_embeddings.append((idx, embedding))
                        self._memory_cache[cache_key] = embedding
                    else:
                        texts_to_encode.append(text)
                        text_indices.append(idx)
            
            # Encode remaining texts
            if texts_to_encode:
                logger.info(f"Encoding {len(texts_to_encode)} new texts")
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
                
                # Cache new embeddings
                for text, embedding in zip(texts_to_encode, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self._memory_cache[cache_key] = embedding
                    self._save_to_disk_cache(cache_key, embedding)
                
                # Combine cached and new
                all_embeddings = [None] * len(texts)
                for idx, emb in cached_embeddings:
                    all_embeddings[idx] = emb
                for idx, emb in zip(text_indices, new_embeddings):
                    all_embeddings[idx] = emb
                
                return np.array(all_embeddings)
            
            else:
                # All cached
                logger.info(f"All {len(texts)} embeddings loaded from cache")
                all_embeddings = [None] * len(texts)
                for idx, emb in cached_embeddings:
                    all_embeddings[idx] = emb
                return np.array(all_embeddings)
        
        else:
            # No caching
            logger.info(f"Encoding {len(texts)} texts (caching disabled)")
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                return joblib.load(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        return None
    
    def _save_to_disk_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            joblib.dump(embedding, cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def clear_cache(self):
        """Clear all caches."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")