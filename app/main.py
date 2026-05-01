"""
Zendesk RAG Bot — Main FastAPI Application

Endpoints:
  GET  /health                  — health check
  POST /ingest/sops             — ingest SOPs into Pinecone
  POST /ingest/tickets          — ingest resolved tickets into Pinecone
  POST /ingest/all              — ingest everything
  POST /query                   — manually query the bot
  POST /webhook/zendesk         — receives NEW tickets from Zendesk
  POST /webhook/ticket-closed   — receives CLOSED tickets from Zendesk (NEW)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import (
    TicketWebhookPayload,
    ClosedTicketWebhookPayload,
    IngestRequest,
    QueryRequest,
)
from app.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
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
    description="Self-improving RAG chatbot using Zendesk SOPs + resolved ticket summaries.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Ingestion endpoints ───────────────────────────────────────────────────────

@app.post("/ingest/sops")
async def ingest_sops():
    """Fetch SOP articles, embed them, store in Pinecone."""
    try:
        result = await rag.ingest_sops()
        return {"status": "success", **result}
    except Exception as e:
        logger.exception("SOP ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/tickets")
async def ingest_tickets():
    """Fetch closed Zendesk tickets, embed summaries, store in Pinecone."""
    try:
        result = await rag.ingest_tickets()
        return {"status": "success", **result}
    except Exception as e:
        logger.exception("Ticket ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/all")
async def ingest_all():
    """Full ingestion: SOPs + resolved tickets."""
    try:
        sop_result = await rag.ingest_sops()
        ticket_result = await rag.ingest_tickets()
        return {
            "status": "success",
            "sops": sop_result,
            "tickets": ticket_result,
        }
    except Exception as e:
        logger.exception("Full ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Manual query ──────────────────────────────────────────────────────────────

@app.post("/query")
async def query(req: QueryRequest):
    """Manually generate a response for a ticket description."""
    try:
        result = await rag.generate_response(
            ticket_description=req.ticket_description,
            ticket_id=req.ticket_id,
        )
        return result
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Webhook: New ticket created ───────────────────────────────────────────────

@app.post("/webhook/zendesk")
async def zendesk_webhook(request: Request):
    """
    Receives new ticket events from Zendesk.
    Generates a response and posts it as an internal note.

    Zendesk trigger body:
    {
      "ticket_id": "{{ticket.id}}",
      "ticket_description": "Subject: {{ticket.title}}\\n\\nDescription: {{ticket.description}}"
    }
    """
    raw_body = await request.body()
    logger.info(f"New ticket webhook received: {raw_body.decode()[:200]}")

    try:
        payload = TicketWebhookPayload.model_validate_json(raw_body)
    except Exception as e:
        logger.error(f"Payload parse error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    try:
        # Generate response
        result = await rag.generate_response(
            ticket_description=payload.ticket_description,
            ticket_id=payload.ticket_id,
        )
        logger.info(f"Response generated for ticket {payload.ticket_id}. Confidence: {result['confidence']}")

        # Post back to Zendesk as internal note
        if settings.AUTO_POST_TO_ZENDESK:
            note = _format_internal_note(result)
            await rag.zendesk_client.post_internal_note(
                ticket_id=payload.ticket_id_int,
                note=note,
            )
            logger.info(f"Posted internal note to ticket {payload.ticket_id}")

        return {
            "status": "processed",
            "ticket_id": payload.ticket_id,
            "confidence": result["confidence"],
        }

    except Exception as e:
        logger.exception(f"Webhook processing failed for ticket {payload.ticket_id}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Webhook: Ticket closed (NEW — self-learning) ──────────────────────────────

@app.post("/webhook/ticket-closed")
async def ticket_closed_webhook(request: Request):
    """
    Receives closed/solved ticket events from Zendesk.
    Automatically:
    1. Fetches full ticket conversation
    2. Generates a structured summary (problem + actions + resolution)
    3. Stores the summary in Pinecone for future retrieval

    This is the self-learning loop — every closed ticket makes the bot smarter.

    Zendesk trigger body:
    {
      "ticket_id": "{{ticket.id}}",
      "event": "closed",
      "subject": "{{ticket.title}}",
      "description": "{{ticket.description}}"
    }
    """
    raw_body = await request.body()
    logger.info(f"Closed ticket webhook received: {raw_body.decode()[:200]}")

    try:
        payload = ClosedTicketWebhookPayload.model_validate_json(raw_body)
    except Exception as e:
        logger.error(f"Closed ticket payload parse error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    try:
        result = await rag.summarise_and_store_closed_ticket(
            ticket_id=payload.ticket_id,
            subject=payload.subject,
            description=payload.description,
        )

        logger.info(
            f"Ticket {payload.ticket_id} summarised. "
            f"Category: {result['category']}, "
            f"Stored: {result['stored_in_pinecone']}"
        )

        return {
            "status": "summary_stored",
            "ticket_id": payload.ticket_id,
            "category": result["category"],
            "tags": result["tags"],
            "problem": result["problem"],
            "resolution": result["resolution"],
        }

    except Exception as e:
        logger.exception(f"Closed ticket processing failed for ticket {payload.ticket_id}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_internal_note(result: dict) -> str:
    """Format the RAG response into a clear internal note for agents."""
    confidence = result.get("confidence", "MEDIUM")
    response = result.get("response", "")

    confidence_label = {
        "HIGH": "✅ High confidence — based on matching SOP",
        "MEDIUM": "⚠️ Medium confidence — review before sending",
        "MANUAL_REVIEW": "🔴 Manual review required — no matching SOP found",
    }.get(confidence, "⚠️ Review before sending")

    return f"""🤖 RAG Bot Suggestion
{confidence_label}

{response}"""


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
