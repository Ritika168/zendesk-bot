"""
Embedding service — uses sentence-transformers locally (no API cost).
Model: all-MiniLM-L6-v2  (384-dim, ~80MB, fast on CPU)
"""

import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        logger.info(f"Loading embedding model '{settings.EMBEDDING_MODEL}'…")
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Batch embed multiple texts efficiently."""
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return [v.tolist() for v in vectors]
