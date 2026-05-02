"""
Zendesk RAG Bot v2.1 — Main FastAPI Application
Fixes: richer closed ticket note with actions shown
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import (
    TicketWebhookPayload,
    ClosedTicketWebhookPayload,
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
    version="2.1.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.1.0"}


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


# ── Webhook: New ticket ───────────────────────────────────────────────────────

@app.post("/webhook/zendesk")
async def zendesk_webhook(request: Request):
    """
    Receives NEW ticket from Zendesk trigger.
    Generates a suggested response and posts as internal note.

    Trigger body:
    {
      "ticket_id": "{{ticket.id}}",
      "ticket_description": "Subject: {{ticket.title}}\n\nDescription: {{ticket.description}}"
    }
    """
    raw_body = await request.body()
    logger.info(f"New ticket webhook: {raw_body.decode()[:300]}")

    try:
        payload = TicketWebhookPayload.model_validate_json(raw_body)
    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    try:
        result = await rag.generate_response(
            ticket_description=payload.ticket_description,
            ticket_id=payload.ticket_id,
        )
        logger.info(f"Response generated. Confidence: {result['confidence']}")

        if settings.AUTO_POST_TO_ZENDESK:
            note = _format_new_ticket_note(result)
            await rag.zendesk_client.post_internal_note(
                ticket_id=payload.ticket_id_int,
                note=note,
            )

        return {"status": "processed", "ticket_id": payload.ticket_id, "confidence": result["confidence"]}

    except Exception as e:
        logger.exception(f"Webhook failed for ticket {payload.ticket_id}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Webhook: Ticket closed ────────────────────────────────────────────────────

@app.post("/webhook/ticket-closed")
async def ticket_closed_webhook(request: Request):
    """
    Receives SOLVED ticket from Zendesk trigger.
    Summarises, categorises, and stores in Pinecone.
    Posts a summary note on the ticket.

    DUPLICATE PREVENTION: Same ticket ID always maps to same vector ID.
    Pinecone upsert overwrites → no duplicates ever created.

    Trigger body:
    {
      "ticket_id": "{{ticket.id}}",
      "event": "closed",
      "subject": "{{ticket.title}}",
      "description": "{{ticket.description}}"
    }
    """
    raw_body = await request.body()
    logger.info(f"Closed ticket webhook: {raw_body.decode()[:300]}")

    try:
        payload = ClosedTicketWebhookPayload.model_validate_json(raw_body)
    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    try:
        result = await rag.summarise_and_store_closed_ticket(
            ticket_id=payload.ticket_id,
            subject=payload.subject,
            description=payload.description,
        )

        logger.info(
            f"Ticket {payload.ticket_id} stored. "
            f"Category: {result['category']}, "
            f"Duplicate: {result['was_duplicate']}"
        )

        if settings.AUTO_POST_TO_ZENDESK:
            note = _format_closed_ticket_note(result)
            await rag.zendesk_client.post_internal_note(
                ticket_id=payload.ticket_id_int,
                note=note,
            )
            logger.info(f"Posted summary note to ticket {payload.ticket_id}")

        return {
            "status": "summary_stored",
            "ticket_id": payload.ticket_id,
            "category": result["category"],
            "tags": result["tags"],
            "problem": result["problem"],
            "resolution": result["resolution"],
            "stored_in_pinecone": result["stored_in_pinecone"],
            "was_duplicate": result["was_duplicate"],
        }

    except Exception as e:
        logger.exception(f"Closed ticket failed for {payload.ticket_id}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Note formatters ───────────────────────────────────────────────────────────

def _format_new_ticket_note(result: dict) -> str:
    confidence = result.get("confidence", "MEDIUM")
    response = result.get("response", "")
    retrieved_sops = result.get("retrieved_sops", [])
    retrieved_tickets = result.get("retrieved_tickets", [])

    confidence_line = {
        "HIGH":          "✅ HIGH confidence — strong SOP match found",
        "MEDIUM":        "⚠️  MEDIUM confidence — review before sending",
        "MANUAL_REVIEW": "🔴 MANUAL REVIEW REQUIRED — no matching SOP found",
    }.get(confidence, "⚠️  Review before sending")

    sop_lines = ""
    if retrieved_sops:
        sop_lines = "\n\n📋 SOPs used:\n"
        for s in retrieved_sops:
            sop_lines += f"  • {s['title']} (score: {s['score']})\n"

    ticket_lines = ""
    if retrieved_tickets:
        ticket_lines = "\n🎫 Similar past tickets used:\n"
        for t in retrieved_tickets:
            ticket_lines += f"  • {t['title']} (score: {t['score']})\n"
    else:
        ticket_lines = "\n🎫 Similar past tickets: None yet (will improve over time)"

    return (
        f"🤖 RAG Bot Suggestion\n"
        f"{confidence_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{response}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📚 Knowledge sources:{sop_lines}{ticket_lines}"
    )


def _format_closed_ticket_note(result: dict) -> str:
    stored = (
        "✅ Stored — will be used for future similar tickets"
        if result.get("stored_in_pinecone")
        else "❌ Storage failed"
    )
    duplicate_note = (
        "\n⚠️  Note: This ticket was already in the knowledge base — updated with latest summary."
        if result.get("was_duplicate")
        else ""
    )

    actions = result.get("actions", "Not recorded")
    # Format bullet points nicely
    if actions and not actions.startswith("-"):
        actions = "- " + actions

    return (
        f"📦 RAG Bot — Ticket Knowledge Stored\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🔍 PROBLEM:\n{result.get('problem', 'N/A')}\n\n"
        f"🛠️  ACTIONS TAKEN:\n{actions}\n\n"
        f"✅ RESOLUTION:\n{result.get('resolution', 'N/A')}\n\n"
        f"🏷️  CATEGORY: {result.get('category', 'other').upper()}\n"
        f"🔖 TAGS: {result.get('tags', 'N/A')}\n\n"
        f"💾 Pinecone: {stored}{duplicate_note}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Future tickets similar to this will use this summary as reference."
    )


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
