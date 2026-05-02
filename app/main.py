"""
Zendesk RAG Bot v2.2
KEY FIX: Single smart webhook endpoint that handles BOTH new and closed tickets.
Detects payload type automatically — no need for separate webhook URLs.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import QueryRequest
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
    version="2.2.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.2.0"}


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


# ── SINGLE SMART WEBHOOK — handles both new and closed tickets ────────────────

@app.post("/webhook/zendesk")
async def zendesk_webhook(request: Request):
    """
    Single webhook endpoint that handles ALL Zendesk events.

    Detects payload type automatically:
    - If payload has 'event: closed' → summarise and store ticket
    - Otherwise → generate response for new ticket

    This means you only need ONE webhook in Zendesk pointing to this URL.
    Both triggers (new ticket + ticket closed) use the same webhook.

    New ticket trigger body:
    {
      "ticket_id": "{{ticket.id}}",
      "ticket_description": "Subject: {{ticket.title}}\n\nDescription: {{ticket.description}}"
    }

    Closed ticket trigger body:
    {
      "ticket_id": "{{ticket.id}}",
      "event": "closed",
      "subject": "{{ticket.title}}",
      "description": "{{ticket.description}}"
    }
    """
    raw_body = await request.body()
    logger.info(f"Webhook received: {raw_body.decode()[:300]}")

    try:
        import json
        data = json.loads(raw_body)
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}")

    ticket_id = str(data.get("ticket_id", ""))
    event = data.get("event", "new")

    if not ticket_id:
        raise HTTPException(status_code=422, detail="Missing ticket_id")

    # ── Route: Closed ticket → summarise and store ────────────────────────────
    if event == "closed":
        logger.info(f"Routing ticket {ticket_id} to CLOSED handler")
        try:
            result = await rag.summarise_and_store_closed_ticket(
                ticket_id=ticket_id,
                subject=data.get("subject", ""),
                description=data.get("description", ""),
            )

            logger.info(
                f"Ticket {ticket_id} summarised. "
                f"Category: {result['category']}, "
                f"Stored: {result['stored_in_pinecone']}, "
                f"Duplicate: {result['was_duplicate']}"
            )

            if settings.AUTO_POST_TO_ZENDESK:
                note = _format_closed_ticket_note(result)
                await rag.zendesk_client.post_internal_note(
                    ticket_id=int(ticket_id),
                    note=note,
                )
                logger.info(f"Posted summary note to ticket {ticket_id}")

            return {
                "status": "summary_stored",
                "ticket_id": ticket_id,
                "category": result["category"],
                "tags": result["tags"],
                "problem": result["problem"],
                "resolution": result["resolution"],
                "stored_in_pinecone": result["stored_in_pinecone"],
                "was_duplicate": result["was_duplicate"],
            }

        except Exception as e:
            logger.exception(f"Closed ticket processing failed for {ticket_id}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── Route: New ticket → generate response ─────────────────────────────────
    else:
        ticket_description = data.get("ticket_description", "")
        if not ticket_description:
            # Fallback: build description from subject + description fields
            subject = data.get("subject", "")
            description = data.get("description", "")
            ticket_description = f"Subject: {subject}\n\nDescription: {description}"

        logger.info(f"Routing ticket {ticket_id} to NEW TICKET handler")

        try:
            result = await rag.generate_response(
                ticket_description=ticket_description,
                ticket_id=ticket_id,
            )
            logger.info(f"Response generated. Confidence: {result['confidence']}")

            if settings.AUTO_POST_TO_ZENDESK:
                note = _format_new_ticket_note(result)
                await rag.zendesk_client.post_internal_note(
                    ticket_id=int(ticket_id),
                    note=note,
                )
                logger.info(f"Posted internal note to ticket {ticket_id}")

            return {
                "status": "processed",
                "ticket_id": ticket_id,
                "confidence": result["confidence"],
            }

        except Exception as e:
            logger.exception(f"New ticket processing failed for {ticket_id}")
            raise HTTPException(status_code=500, detail=str(e))


# ── Keep the dedicated closed endpoint too (for manual curl testing) ──────────

@app.post("/webhook/ticket-closed")
async def ticket_closed_webhook(request: Request):
    """Dedicated endpoint for manual testing of closed ticket flow."""
    raw_body = await request.body()
    logger.info(f"Closed ticket webhook: {raw_body.decode()[:300]}")

    try:
        import json
        data = json.loads(raw_body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}")

    ticket_id = str(data.get("ticket_id", ""))
    if not ticket_id:
        raise HTTPException(status_code=422, detail="Missing ticket_id")

    try:
        result = await rag.summarise_and_store_closed_ticket(
            ticket_id=ticket_id,
            subject=data.get("subject", ""),
            description=data.get("description", ""),
        )

        if settings.AUTO_POST_TO_ZENDESK:
            note = _format_closed_ticket_note(result)
            await rag.zendesk_client.post_internal_note(
                ticket_id=int(ticket_id),
                note=note,
            )

        return {
            "status": "summary_stored",
            "ticket_id": ticket_id,
            "category": result["category"],
            "tags": result["tags"],
            "problem": result["problem"],
            "resolution": result["resolution"],
            "stored_in_pinecone": result["stored_in_pinecone"],
            "was_duplicate": result["was_duplicate"],
        }

    except Exception as e:
        logger.exception(f"Closed ticket failed for {ticket_id}")
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

    ticket_lines = "\n🎫 Similar past tickets: None yet"
    if retrieved_tickets:
        ticket_lines = "\n🎫 Similar past tickets used:\n"
        for t in retrieved_tickets:
            ticket_lines += f"  • {t['title']} (score: {t['score']})\n"

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
        "\n⚠️  Already existed — updated with latest summary."
        if result.get("was_duplicate")
        else "\n🆕 New entry added to knowledge base."
    )

    actions = result.get("actions", "Not recorded")
    if actions and not actions.strip().startswith("-"):
        actions_formatted = "- " + actions
    else:
        actions_formatted = actions

    return (
        f"📦 RAG Bot — Ticket Knowledge Stored\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🔍 PROBLEM:\n{result.get('problem', 'N/A')}\n\n"
        f"🛠️  ACTIONS TAKEN:\n{actions_formatted}\n\n"
        f"✅ RESOLUTION:\n{result.get('resolution', 'N/A')}\n\n"
        f"🏷️  CATEGORY: {result.get('category', 'other').upper()}\n"
        f"🔖 TAGS: {result.get('tags', 'N/A')}\n\n"
        f"💾 Pinecone: {stored}{duplicate_note}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Future similar tickets will use this as reference."
    )


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
