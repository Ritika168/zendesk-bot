"""
Embedding service — HuggingFace Inference API (free tier).
Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
No local model, no RAM spike, completely free.
"""

import logging
import os
import httpx

logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self):
        token = os.environ.get("HF_API_TOKEN", "")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = httpx.Client(timeout=60.0)
        logger.info("EmbeddingService ready (HuggingFace API, all-MiniLM-L6-v2, 384-dim).")

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = self._client.post(
                HF_API_URL,
                headers=self._headers,
                json={"inputs": batch, "options": {"wait_for_model": True}},
            )
            resp.raise_for_status()
            result = resp.json()
            # HF returns list of embeddings directly
            all_embeddings.extend(result)
            logger.info(f"Embedded {min(i+batch_size, len(texts))}/{len(texts)} texts...")
        return all_embeddings
