"""
Zendesk RAG Bot v2.0 — Main FastAPI Application
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
    description="Self-improving RAG chatbot using SOPs + resolved ticket summaries.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest/sops")
async def ingest_sops():
    try:
        result = await rag.ingest_sops()
        return {"status": "success", **result}
    except Exception as e:
        logger.exception("SOP ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/tickets")
async def ingest_tickets():
    try:
        result = await rag.ingest_tickets()
        return {"status": "success", **result}
    except Exception as e:
        logger.exception("Ticket ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/all")
async def ingest_all():
    try:
        sop_result = await rag.ingest_sops()
        ticket_result = await rag.ingest_tickets()
        return {"status": "success", "sops": sop_result, "tickets": ticket_result}
    except Exception as e:
        logger.exception("Full ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Manual query ──────────────────────────────────────────────────────────────

@app.post("/query")
async def query(req: QueryRequest):
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
    Receives NEW ticket from Zendesk.
    Generates a suggested response and posts it as an internal note.

    Zendesk trigger body should be:
    {
      "ticket_id": "{{ticket.id}}",
      "ticket_description": "Subject: {{ticket.title}}\n\nDescription: {{ticket.description}}"
    }
    """
    raw_body = await request.body()
    logger.info(f"New ticket webhook received: {raw_body.decode()[:300]}")

    try:
        payload = TicketWebhookPayload.model_validate_json(raw_body)
    except Exception as e:
        logger.error(f"Payload parse error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    try:
        result = await rag.generate_response(
            ticket_description=payload.ticket_description,
            ticket_id=payload.ticket_id,
        )

        logger.info(f"Response generated for ticket {payload.ticket_id}. Confidence: {result['confidence']}")

        if settings.AUTO_POST_TO_ZENDESK:
            note = _format_new_ticket_note(result)
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


# ── Webhook: Ticket closed — self-learning loop ───────────────────────────────

@app.post("/webhook/ticket-closed")
async def ticket_closed_webhook(request: Request):
    """
    Receives SOLVED ticket from Zendesk.

    Automatically:
    1. Fetches full conversation from Zendesk
    2. Generates structured summary: problem + actions + resolution + category + tags
    3. Stores the summary in Pinecone (type=TICKET)
    4. Posts a summary note on the ticket so agents can see what was stored

    This is the self-learning loop — every closed ticket improves future responses.

    Zendesk trigger body:
    {
      "ticket_id": "{{ticket.id}}",
      "event": "closed",
      "subject": "{{ticket.title}}",
      "description": "{{ticket.description}}"
    }
    """
    raw_body = await request.body()
    logger.info(f"Closed ticket webhook received: {raw_body.decode()[:300]}")

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
            f"Stored in Pinecone: {result['stored_in_pinecone']}"
        )

        # Post a summary note on the ticket so agents can see what was stored
        if settings.AUTO_POST_TO_ZENDESK:
            note = _format_closed_ticket_note(result)
            await rag.zendesk_client.post_internal_note(
                ticket_id=payload.ticket_id_int,
                note=note,
            )
            logger.info(f"Posted summary note to closed ticket {payload.ticket_id}")

        return {
            "status": "summary_stored",
            "ticket_id": payload.ticket_id,
            "category": result["category"],
            "tags": result["tags"],
            "problem": result["problem"],
            "resolution": result["resolution"],
            "stored_in_pinecone": result["stored_in_pinecone"],
        }

    except Exception as e:
        logger.exception(f"Closed ticket processing failed for ticket {payload.ticket_id}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Note formatters ───────────────────────────────────────────────────────────

def _format_new_ticket_note(result: dict) -> str:
    """
    Format the RAG response as a clear internal note for agents.
    Shows: confidence, suggested reply, and which SOPs/tickets were used.
    """
    confidence = result.get("confidence", "MEDIUM")
    response = result.get("response", "")
    retrieved_sops = result.get("retrieved_sops", [])
    retrieved_tickets = result.get("retrieved_tickets", [])

    confidence_line = {
        "HIGH":          "✅ HIGH confidence — strong SOP match found",
        "MEDIUM":        "⚠️  MEDIUM confidence — review before sending",
        "MANUAL_REVIEW": "🔴 MANUAL REVIEW REQUIRED — no matching SOP found",
    }.get(confidence, "⚠️  Review before sending")

    # List which SOPs were used
    sop_lines = ""
    if retrieved_sops:
        sop_lines = "\n📋 SOPs used:\n"
        for s in retrieved_sops:
            sop_lines += f"  • {s['title']} (score: {s['score']})\n"

    # List which past ticket summaries were used
    ticket_lines = ""
    if retrieved_tickets:
        ticket_lines = "\n🎫 Similar past tickets used:\n"
        for t in retrieved_tickets:
            ticket_lines += f"  • {t['title']} (score: {t['score']})\n"

    return f"""🤖 RAG Bot Suggestion
{confidence_line}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{response}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 Knowledge sources:{sop_lines}{ticket_lines if ticket_lines else chr(10) + '  No similar past tickets found yet.'}"""


def _format_closed_ticket_note(result: dict) -> str:
    """
    Format the ticket summary as an internal note on the closed ticket.
    This shows agents exactly what was stored in Pinecone for future use.
    """
    stored = "✅ Yes — will be used for future similar tickets" if result.get("stored_in_pinecone") else "❌ Storage failed"

    return f"""📦 RAG Bot — Ticket Summary Stored
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This ticket has been summarised and added to the knowledge base.

🔍 PROBLEM:
{result.get('problem', 'N/A')}

✅ RESOLUTION:
{result.get('resolution', 'N/A')}

🏷️  CATEGORY: {result.get('category', 'other').upper()}
🔖 TAGS: {result.get('tags', 'N/A')}

💾 Stored in Pinecone: {stored}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Future tickets similar to this one will automatically use this summary as reference."""


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
