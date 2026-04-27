"""
RAG Pipeline — orchestrates ingestion and query/response generation.
This is the single integration point that wires together:
  ZendeskClient → preprocessor → EmbeddingService → VectorStore
  VectorStore + LLMClient → response
"""

import logging
import asyncio
from app.zendesk_client import ZendeskClient
from app.preprocessor import preprocess_sop_article, preprocess_ticket
from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.llm_client import LLMClient
from app.models import RetrievedChunk, RAGResponse
from app.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.zendesk_client = ZendeskClient()
        self.embedder = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm = LLMClient()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def ingest_sops(self, limit: int = 200) -> dict:
        """Fetch → preprocess → embed → upsert SOP articles."""
        articles = await self.zendesk_client.fetch_sop_articles(limit=limit)
        logger.info(f"Processing {len(articles)} SOP articles…")

        all_chunks: list[dict] = []
        for article in articles:
            all_chunks.extend(preprocess_sop_article(article))

        if not all_chunks:
            return {"articles_fetched": len(articles), "chunks_upserted": 0}

        # Batch embed
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb

        upserted = self.vector_store.upsert_chunks(all_chunks)
        logger.info(f"SOP ingestion complete. Upserted {upserted} chunks.")
        return {"articles_fetched": len(articles), "chunks_upserted": upserted}

    async def ingest_tickets(self, limit: int = 500) -> dict:
        """Fetch closed tickets → extract summaries → embed → upsert."""
        tickets = await self.zendesk_client.fetch_closed_tickets(limit=limit)
        logger.info(f"Processing {len(tickets)} resolved tickets…")

        all_chunks: list[dict] = []

        # Fetch comments concurrently in small batches to avoid rate limits
        async def process_ticket(ticket: dict):
            try:
                comments = await self.zendesk_client.get_ticket_comments(ticket["id"])
                return preprocess_ticket(ticket, comments)
            except Exception as e:
                logger.warning(f"Skipping ticket {ticket.get('id')}: {e}")
                return []

        # Process 10 tickets concurrently
        for i in range(0, len(tickets), 10):
            batch = tickets[i : i + 10]
            results = await asyncio.gather(*[process_ticket(t) for t in batch])
            for chunks in results:
                all_chunks.extend(chunks)

        if not all_chunks:
            return {"tickets_fetched": len(tickets), "chunks_upserted": 0}

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb

        upserted = self.vector_store.upsert_chunks(all_chunks)
        logger.info(f"Ticket ingestion complete. Upserted {upserted} chunks.")
        return {"tickets_fetched": len(tickets), "chunks_upserted": upserted}

    # ── Query / Response generation ───────────────────────────────────────────

    async def generate_response(
        self,
        ticket_description: str,
        ticket_id: str | None = None,
    ) -> dict:
        """
        Full RAG query flow:
        1. Embed the ticket description
        2. Query Pinecone for SOPs and similar tickets separately
        3. Generate LLM response
        4. Return structured result
        """
        logger.info(f"Generating response for ticket {ticket_id or 'ad-hoc'}…")

        # Step 1 — embed query
        query_embedding = self.embedder.embed(ticket_description)

        # Step 2 — retrieve from both namespaces in parallel
        sop_hits, ticket_hits = await asyncio.gather(
            asyncio.to_thread(
                self.vector_store.query,
                query_embedding,
                filter_type="SOP",
                top_k=settings.TOP_K_RESULTS,
            ),
            asyncio.to_thread(
                self.vector_store.query,
                query_embedding,
                filter_type="TICKET",
                top_k=settings.TOP_K_RESULTS,
            ),
        )

        logger.info(f"Retrieved {len(sop_hits)} SOP chunks, {len(ticket_hits)} ticket chunks.")

        # Step 3 — generate LLM response
        response_text, confidence = await self.llm.generate(
            ticket_description=ticket_description,
            sop_chunks=sop_hits,
            ticket_chunks=ticket_hits,
        )

        # Step 4 — build structured response
        return RAGResponse(
            ticket_id=ticket_id,
            response=response_text,
            retrieved_sops=[RetrievedChunk(**h) for h in sop_hits],
            retrieved_tickets=[RetrievedChunk(**h) for h in ticket_hits],
            confidence=confidence,
        ).model_dump()
