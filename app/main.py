import logging
from contextlib import asynccontextmanager
import hmac, hashlib

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse

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


app = FastAPI(title="Zendesk RAG Bot", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


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


@app.post("/webhook/zendesk")
async def zendesk_webhook(request: Request):
    raw_body = await request.body()
    logger.info(f"Webhook received: {raw_body.decode()}")

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
        logger.info(f"Generated response for ticket {payload.ticket_id}")

        if settings.AUTO_POST_TO_ZENDESK:
            await rag.zendesk_client.post_internal_note(
                ticket_id=int(payload.ticket_id),
                note=result["response"],
            )
            logger.info(f"Posted internal note to ticket {payload.ticket_id}")

        return {"status": "processed", "ticket_id": payload.ticket_id}
    except Exception as e:
        logger.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
