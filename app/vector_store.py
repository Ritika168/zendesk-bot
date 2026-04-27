"""
Pinecone vector store — upsert and query operations.

Index schema
────────────
Dimension : 384  (all-MiniLM-L6-v2)
Metric    : cosine
Pod type  : starter (free tier)

Metadata fields stored per vector
──────────────────────────────────
  type       : "SOP" | "TICKET"
  source_id  : original Zendesk article/ticket ID
  title      : human-readable title
  url        : article URL (SOP only)
  chunk_index: position within source document
  text       : raw chunk text (stored for retrieval)
"""

import logging
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 100  # Pinecone upsert batch limit


class VectorStore:
    def __init__(self):
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = self._get_or_create_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _get_or_create_index(self):
        existing = [idx.name for idx in self._pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating Pinecone index '{settings.PINECONE_INDEX_NAME}'…")
            self._pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # free tier
            )
            logger.info("Index created.")
        else:
            logger.info(f"Using existing Pinecone index '{settings.PINECONE_INDEX_NAME}'.")

        return self._pc.Index(settings.PINECONE_INDEX_NAME)

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """
        Upsert a list of chunk dicts.
        Each dict must have: id, embedding (list[float]), metadata (dict).
        Returns total vectors upserted.
        """
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vectors = [
                {
                    "id": c["id"],
                    "values": c["embedding"],
                    "metadata": {**c["metadata"], "text": c["text"]},
                }
                for c in batch
            ]
            self._index.upsert(vectors=vectors)
            total += len(batch)
            logger.info(f"Upserted {total}/{len(chunks)} vectors…")

        return total

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        filter_type: str | None = None,
        top_k: int = settings.TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Query Pinecone for nearest neighbours.
        Optionally filter by metadata 'type' ("SOP" or "TICKET").
        Returns list of match dicts with score + metadata.
        """
        filter_dict = {"type": {"$eq": filter_type}} if filter_type else None

        response = self._index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )

        results = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append(
                {
                    "id": match["id"],
                    "score": round(match["score"], 4),
                    "type": meta.get("type", ""),
                    "source_id": meta.get("source_id", ""),
                    "title": meta.get("title", ""),
                    "text": meta.get("text", ""),
                }
            )
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return self._index.describe_index_stats()
