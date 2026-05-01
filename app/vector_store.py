"""
Pinecone vector store — upsert and query operations.

Index schema
────────────
Dimension : 384  (all-MiniLM-L6-v2 via HuggingFace)
Metric    : cosine
Spec      : serverless, aws us-east-1 (free tier)

Metadata fields stored per vector
──────────────────────────────────
  type       : "SOP" | "TICKET"
  source_id  : original Zendesk article/ticket ID
  title      : human-readable title
  category   : billing | authentication | technical | account | refund | other
  tags       : comma-separated keywords
  chunk_index: position within source document
  text       : raw chunk text (stored for retrieval)
"""

import logging
from pinecone import Pinecone, ServerlessSpec
from app.config import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class VectorStore:
    def __init__(self):
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = self._get_or_create_index()

    def _get_or_create_index(self):
        existing = [idx.name for idx in self._pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating Pinecone index '{settings.PINECONE_INDEX_NAME}'...")
            self._pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Index created.")
        else:
            logger.info(f"Using existing Pinecone index '{settings.PINECONE_INDEX_NAME}'.")
        return self._pc.Index(settings.PINECONE_INDEX_NAME)

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """
        Upsert chunks into Pinecone.
        Each chunk must have: id, embedding, text, metadata (dict).
        """
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i: i + BATCH_SIZE]
            vectors = [
                {
                    "id": c["id"],
                    "values": c["embedding"],
                    "metadata": {
                        **c["metadata"],
                        "text": c["text"],
                    },
                }
                for c in batch
            ]
            self._index.upsert(vectors=vectors)
            total += len(batch)
            logger.info(f"Upserted {total}/{len(chunks)} vectors...")

        return total

    def upsert_ticket_summary(self, chunk: dict) -> bool:
        """
        Upsert a single ticket summary chunk.
        Used by the closed-ticket webhook.
        chunk must have: id, embedding, text, metadata
        """
        try:
            self._index.upsert(vectors=[
                {
                    "id": chunk["id"],
                    "values": chunk["embedding"],
                    "metadata": {
                        **chunk["metadata"],
                        "text": chunk["text"],
                    },
                }
            ])
            logger.info(f"Upserted ticket summary: {chunk['id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert ticket summary: {e}")
            return False

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        filter_type: str | None = None,
        filter_category: str | None = None,
        top_k: int = settings.TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Query Pinecone for nearest neighbours.
        Optionally filter by:
          - filter_type: "SOP" or "TICKET"
          - filter_category: "billing", "authentication", etc.
        """
        # Build filter
        filter_dict = {}
        if filter_type:
            filter_dict["type"] = {"$eq": filter_type}
        if filter_category and filter_category != "other":
            filter_dict["category"] = {"$eq": filter_category}

        response = self._index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
        )

        results = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append({
                "id": match["id"],
                "score": round(match["score"], 4),
                "type": meta.get("type", ""),
                "source_id": meta.get("source_id", ""),
                "title": meta.get("title", ""),
                "category": meta.get("category", "other"),
                "tags": meta.get("tags", ""),
                "text": meta.get("text", ""),
            })
        return results

    def stats(self) -> dict:
        return self._index.describe_index_stats()
