"""
Embedding service — HuggingFace Inference API (free tier).
Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
"""

import logging
import os
import httpx

logger = logging.getLogger(__name__)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


class EmbeddingService:
    def __init__(self):
        token = os.environ.get("HF_API_TOKEN", "")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=60.0)
        logger.info("EmbeddingService ready (HuggingFace, all-MiniLM-L6-v2, 384-dim).")

    def embed(self, text: str) -> list[float]:
        resp = self._client.post(
            HF_API_URL,
            headers=self._headers,
            json={"inputs": text},
        )
        resp.raise_for_status()
        result = resp.json()
        # Returns nested list for single input — flatten if needed
        if isinstance(result[0], list):
            return result[0]
        return result

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            for text in batch:
                all_embeddings.append(self.embed(text))
            logger.info(f"Embedded {min(i+batch_size, len(texts))}/{len(texts)} texts...")
        return all_embeddings
