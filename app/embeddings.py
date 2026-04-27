    

import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        logger.info("EmbeddingService ready (Groq API, nomic-embed-text-v1_5).")

    def embed(self, text: str) -> list[float]:
        resp = self._client.post(
            "https://api.groq.com/openai/v1/embeddings",
            json={"model": "nomic-embed-text-v1_5", "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            for text in batch:
                all_embeddings.append(self.embed(text))
            logger.info(f"Embedded {min(i+batch_size, len(texts))}/{len(texts)} texts...")
        return all_embeddings
