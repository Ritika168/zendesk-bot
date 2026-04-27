"""
Zendesk RAG Bot — Main FastAPI Application
Handles webhook events and ticket response generation.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
import hmac, hashlib

from app.config import settings
from app.models import TicketWebhookPayload, IngestRequest, QueryRequest
from app.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

rag: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Zendesk RAG Bot",
    description="Cloud RAG chatbot using Zendesk SOPs + resolved ticket summaries.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def verify_zendesk_signature(raw_body: bytes, signature: str) -> bool:
    """Validate Zendesk webhook HMAC-SHA256 signature."""
    if not settings.ZENDESK_WEBHOOK_SECRET:
        return True  # Skip validation if secret not configured
    expected = hmac.new(
        settings.ZENDESK_WEBHOOK_SECRET.encode(),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature or "")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Ingest endpoints ──────────────────────────────────────────────────────────

@app.post("/ingest/sops")
async def ingest_sops(req: IngestRequest | None = None):
    """
    Fetch all SOP articles from Zendesk Help Center,
    embed them, and upsert into Pinecone.
    """
    try:
        result = await rag.ingest_sops(limit=getattr(req, "limit", 200) if req else 200)
        return {"status": "success", **result}
    except Exception as e:
        logger.exception("SOP ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/tickets")
async def ingest_tickets(req: IngestRequest | None = None):
    """
    Fetch closed Zendesk tickets, extract summaries,
    embed them, and upsert into Pinecone.
    """
    try:
        result = await rag.ingest_tickets(limit=getattr(req, "limit", 500) if req else 500)
        return {"status": "success", **result}
    except Exception as e:
        logger.exception("Ticket ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/all")
async def ingest_all():
    """Run full ingestion: SOPs + resolved tickets."""
    try:
        sop_result = await rag.ingest_sops()
        ticket_result = await rag.ingest_tickets()
        return {"status": "success", "sops": sop_result, "tickets": ticket_result}
    except Exception as e:
        logger.exception("Full ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Query endpoint ────────────────────────────────────────────────────────────

@app.post("/query")
async def query(req: QueryRequest):
    """
    Accepts a ticket description and returns a suggested response
    generated from SOP + past ticket summaries.
    """
    try:
        result = await rag.generate_response(
            ticket_description=req.ticket_description,
            ticket_id=req.ticket_id,
        )
        return result
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Zendesk Webhook ───────────────────────────────────────────────────────────

@app.post("/webhook/zendesk")
async def zendesk_webhook(
    request: Request,
    x_zendesk_webhook_signature: str = Header(default=""),
):
    """
    Receives new ticket events from Zendesk via webhook trigger.
    Generates a response and posts it back as an internal note.
    """
    raw_body = await request.body()

    if not verify_zendesk_signature(raw_body, x_zendesk_webhook_signature):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    try:
        payload = TicketWebhookPayload.model_validate_json(raw_body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    try:
        result = await rag.generate_response(
            ticket_description=payload.ticket_description,
            ticket_id=str(payload.ticket_id),
        )

        # Post suggestion back to Zendesk as an internal note
        if settings.AUTO_POST_TO_ZENDESK:
            await rag.zendesk_client.post_internal_note(
                ticket_id=payload.ticket_id,
                note=result["response"],
            )

        return {"status": "processed", "ticket_id": payload.ticket_id}
    except Exception as e:
        logger.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
