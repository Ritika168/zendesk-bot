"""
RAG Pipeline v2.1 — orchestrates all ingestion and query/response flows.
Fixes: duplicate prevention via deterministic vector IDs.
"""

import json
import logging
import asyncio
from pathlib import Path

from app.zendesk_client import ZendeskClient
from app.preprocessor import preprocess_sop_article, preprocess_ticket
from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.llm_client import LLMClient
from app.models import RetrievedChunk, RAGResponse
from app.config import settings

logger = logging.getLogger(__name__)

LOCAL_SOPS_PATH = Path(__file__).parent.parent / "sops.json"


class RAGPipeline:
    def __init__(self):
        self.zendesk_client = ZendeskClient()
        self.embedder = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm = LLMClient()

    # ─────────────────────────────────────────────────────────────────────────
    # INGESTION
    # ─────────────────────────────────────────────────────────────────────────

    async def ingest_sops(self, limit: int = 200) -> dict:
        """Load SOPs from Zendesk Guide or local sops.json. Embed and store."""
        articles = await self.zendesk_client.fetch_sop_articles(limit=limit)

        if not articles and LOCAL_SOPS_PATH.exists():
            logger.info(f"Loading SOPs from local file: {LOCAL_SOPS_PATH}")
            with open(LOCAL_SOPS_PATH, "r") as f:
                raw_sops = json.load(f)
            articles = [
                {
                    "id": sop["id"],
                    "title": sop["title"],
                    "body": sop["body"],
                    "html_url": "",
                }
                for sop in raw_sops
            ]
            logger.info(f"Loaded {len(articles)} SOPs from local file.")

        if not articles:
            logger.warning("No SOPs found.")
            return {"articles_fetched": 0, "chunks_upserted": 0}

        all_chunks = []
        for article in articles:
            chunks = preprocess_sop_article(article)
            for chunk in chunks:
                chunk["metadata"]["category"] = "general"
                chunk["metadata"]["tags"] = ""
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"articles_fetched": len(articles), "chunks_upserted": 0}

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb

        upserted = self.vector_store.upsert_chunks(all_chunks)
        logger.info(f"SOP ingestion complete. Upserted {upserted} chunks.")
        return {"articles_fetched": len(articles), "chunks_upserted": upserted}

    async def ingest_tickets(self, limit: int = 500) -> dict:
        """Fetch closed tickets, extract summaries, embed and store."""
        tickets = await self.zendesk_client.fetch_closed_tickets(limit=limit)
        logger.info(f"Processing {len(tickets)} resolved tickets...")

        all_chunks = []

        async def process_ticket(ticket: dict):
            try:
                comments = await self.zendesk_client.get_ticket_comments(ticket["id"])
                return preprocess_ticket(ticket, comments)
            except Exception as e:
                logger.warning(f"Skipping ticket {ticket.get('id')}: {e}")
                return []

        for i in range(0, len(tickets), 10):
            batch = tickets[i: i + 10]
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

    # ─────────────────────────────────────────────────────────────────────────
    # CLOSED TICKET SUMMARISATION — self-learning loop
    # ─────────────────────────────────────────────────────────────────────────

    async def summarise_and_store_closed_ticket(
        self,
        ticket_id: str,
        subject: str,
        description: str,
    ) -> dict:
        """
        Called when a ticket is marked as solved.
        Generates a structured summary and stores it in Pinecone.

        DUPLICATE PREVENTION:
        Vector ID is always "ticket_summary_{ticket_id}" — Pinecone upsert
        overwrites existing vectors with the same ID, so re-processing the
        same ticket simply updates it rather than creating duplicates.
        """
        logger.info(f"Summarising closed ticket {ticket_id}...")

        # Check if already stored — log it but continue (upsert will overwrite)
        vector_id = f"ticket_summary_{ticket_id}"
        existing = self.vector_store.fetch_by_id(vector_id)
        if existing:
            logger.info(
                f"Ticket {ticket_id} already in Pinecone — will overwrite with latest summary."
            )

        # Fetch full conversation
        try:
            comments = await self.zendesk_client.get_ticket_comments(int(ticket_id))
        except Exception as e:
            logger.warning(f"Could not fetch comments for ticket {ticket_id}: {e}")
            comments = []

        # Generate structured summary
        summary_data = await self.llm.summarise_ticket(
            subject=subject,
            description=description,
            comments=comments,
        )

        # Build text to embed
        embed_text = (
            f"Issue: {subject}\n"
            f"Problem: {summary_data['problem']}\n"
            f"Actions: {summary_data['actions']}\n"
            f"Resolution: {summary_data['resolution']}\n"
            f"Tags: {summary_data['tags']}"
        )

        # Embed
        embedding = self.embedder.embed(embed_text)

        # Build chunk with DETERMINISTIC ID — prevents duplicates
        chunk = {
            "id": vector_id,  # same ticket always gets same ID → overwrites
            "text": embed_text,
            "embedding": embedding,
            "metadata": {
                "type": "TICKET",
                "source_id": ticket_id,
                "title": subject,
                "category": summary_data["category"],
                "tags": summary_data["tags"],
                "problem": summary_data["problem"][:200],
                "resolution": summary_data["resolution"][:200],
            },
        }

        # Upsert — overwrites if already exists
        success = self.vector_store.upsert_ticket_summary(chunk)

        result = {
            "ticket_id": ticket_id,
            "vector_id": vector_id,
            "category": summary_data["category"],
            "tags": summary_data["tags"],
            "problem": summary_data["problem"],
            "actions": summary_data["actions"],
            "resolution": summary_data["resolution"],
            "stored_in_pinecone": success,
            "was_duplicate": existing is not None,
        }

        logger.info(
            f"Ticket {ticket_id} summarised and stored. "
            f"Category: {summary_data['category']}, "
            f"Was duplicate: {existing is not None}"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY — Generate response for new ticket
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_response(
        self,
        ticket_description: str,
        ticket_id: str | None = None,
    ) -> dict:
        """
        Full RAG query flow:
        1. Classify ticket category
        2. Embed the description
        3. Query Pinecone: SOPs + category-filtered past tickets
        4. Fallback to unfiltered if no category hits
        5. Generate LLM response
        """
        logger.info(f"Generating response for ticket {ticket_id or 'ad-hoc'}...")

        # Classify and embed in parallel
        category_task = asyncio.create_task(
            self.llm.classify_ticket(ticket_description)
        )
        query_embedding = self.embedder.embed(ticket_description)
        category = await category_task
        logger.info(f"Ticket classified as: {category}")

        # Query Pinecone
        sop_hits, ticket_hits = await asyncio.gather(
            asyncio.to_thread(
                self.vector_store.query,
                query_embedding,
                "SOP",
                None,
                settings.TOP_K_RESULTS,
            ),
            asyncio.to_thread(
                self.vector_store.query,
                query_embedding,
                "TICKET",
                category,
                settings.TOP_K_RESULTS,
            ),
        )

        # Fallback: no category hits → search all tickets
        if not ticket_hits:
            logger.info("No category-filtered hits, falling back to all tickets...")
            ticket_hits = await asyncio.to_thread(
                self.vector_store.query,
                query_embedding,
                "TICKET",
                None,
                settings.TOP_K_RESULTS,
            )

        logger.info(
            f"Retrieved {len(sop_hits)} SOP chunks, "
            f"{len(ticket_hits)} ticket chunks."
        )

        # Generate response
        response_text, confidence = await self.llm.generate(
            ticket_description=ticket_description,
            sop_chunks=sop_hits,
            ticket_chunks=ticket_hits,
        )

        return RAGResponse(
            ticket_id=ticket_id,
            response=response_text,
            retrieved_sops=[
                RetrievedChunk(**{k: v for k, v in h.items()
                                  if k in RetrievedChunk.model_fields})
                for h in sop_hits
            ],
            retrieved_tickets=[
                RetrievedChunk(**{k: v for k, v in h.items()
                                  if k in RetrievedChunk.model_fields})
                for h in ticket_hits
            ],
            confidence=confidence,
        ).model_dump()
