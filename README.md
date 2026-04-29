# Zendesk RAG Bot 🤖

> A production-ready, cloud-deployed RAG (Retrieval-Augmented Generation) chatbot that automatically generates intelligent support responses on every new Zendesk ticket — using your SOPs as the source of truth.

**Live URL:** `https://zendesk-rag-bot-pqbr.onrender.com`  
**Status:** ✅ Fully operational

---

## What It Does

When a customer creates a support ticket in Zendesk:
1. The bot is instantly triggered via webhook
2. It reads the ticket description
3. Searches your SOP knowledge base for relevant procedures
4. Searches past resolved tickets for similar cases
5. Generates a professional, actionable response using an LLM
6. Posts the suggested response as an **internal note** on the ticket
7. Your agent reviews it and sends it to the customer

All of this happens in **under 3 seconds**, automatically, for every ticket.

---

## Architecture

```
INGESTION PIPELINE (run once / on demand)
─────────────────────────────────────────
sops.json (your SOPs)      Zendesk Closed Tickets
        │                          │
        ▼                          ▼
    Clean + Chunk             Extract Summary
        │                          │
        └──────────┬───────────────┘
                   ▼
     HuggingFace Embeddings API
     (all-MiniLM-L6-v2, 384-dim, free)
                   │
                   ▼
          Pinecone Vector Index
          (type: SOP | TICKET)


QUERY PIPELINE (automatic on every new ticket)
──────────────────────────────────────────────
Customer creates Zendesk ticket
        │
        ▼
Zendesk Trigger fires webhook
        │
        ▼
POST /webhook/zendesk (Render)
        │
        ▼
Embed ticket description (HuggingFace)
        │
        ▼
Query Pinecone ──► Top 3 SOPs + Top 3 past tickets
        │
        ▼
Build prompt (SOP = truth, tickets = reference)
        │
        ▼
Groq LLM (llama-3.1-8b-instant, free)
        │
        ▼
Post internal note → Zendesk ticket ✅
```

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| Backend | FastAPI + Uvicorn | Free |
| Deployment | Render.com | Free tier |
| Embeddings | HuggingFace Inference API (all-MiniLM-L6-v2) | Free |
| Vector DB | Pinecone (serverless, 384-dim, cosine) | Free tier |
| LLM | Groq (llama-3.1-8b-instant) | Free tier |
| Source data | Local sops.json + Zendesk API | Free |

**Total monthly cost: $0**

---

## Project Structure

```
zendesk-rag-bot/
├── app/
│   ├── __init__.py           # Package marker
│   ├── main.py               # FastAPI app + all endpoints + webhook handler
│   ├── config.py             # Environment variable settings (Pydantic)
│   ├── models.py             # Request/response Pydantic models
│   ├── rag_pipeline.py       # Core orchestrator: ingest + query flow
│   ├── zendesk_client.py     # Zendesk API client (tickets + posting notes)
│   ├── preprocessor.py       # HTML cleaning, chunking, summary extraction
│   ├── embeddings.py         # HuggingFace embedding service
│   ├── vector_store.py       # Pinecone upsert + filtered query
│   └── llm_client.py         # Groq LLM client + prompt template
├── scripts/
│   └── ingest.py             # Standalone ingestion CLI
├── tests/
│   └── test_preprocessor.py  # Unit tests
├── sops.json                 # Your SOP knowledge base (edit this!)
├── .env.example              # Environment variable template
├── render.yaml               # Render deployment config
└── requirements.txt          # Python dependencies
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/ingest/sops` | Load SOPs into Pinecone |
| `POST` | `/ingest/tickets` | Load resolved tickets into Pinecone |
| `POST` | `/ingest/all` | Full ingestion (SOPs + tickets) |
| `POST` | `/query` | Manually generate response for a ticket |
| `POST` | `/webhook/zendesk` | Zendesk webhook receiver (auto-triggered) |

---

## Environment Variables

| Variable | Required | Value |
|---|---|---|
| `ZENDESK_SUBDOMAIN` | ✅ | `demo-87927` |
| `ZENDESK_EMAIL` | ✅ | Your Zendesk agent email |
| `ZENDESK_API_TOKEN` | ✅ | Zendesk API token |
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `PINECONE_INDEX_NAME` | ✅ | `zendesk-rag` |
| `GROQ_API_KEY` | ✅ | Groq API key |
| `GROQ_MODEL` | ✅ | `llama-3.1-8b-instant` |
| `HF_API_TOKEN` | ✅ | HuggingFace API token |
| `AUTO_POST_TO_ZENDESK` | ✅ | `true` |
| `TOP_K_RESULTS` | ⬜ | `3` (default) |

---

## Adding Your Own SOPs

Edit `sops.json` in the root of the repo:

```json
[
  {
    "id": "sop-006",
    "title": "Your SOP Title",
    "body": "Step 1: Do this.\nStep 2: Do that.\nStep 3: Escalate if needed."
  }
]
```

After pushing to GitHub, re-run ingestion:
```bash
curl -X POST https://zendesk-rag-bot-pqbr.onrender.com/ingest/sops
```

---

## Quick Test Commands

```bash
# Health check
curl https://zendesk-rag-bot-pqbr.onrender.com/health

# Manual query
curl -X POST https://zendesk-rag-bot-pqbr.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"ticket_description": "I forgot my password and cannot log in"}'

# Re-ingest SOPs
curl -X POST https://zendesk-rag-bot-pqbr.onrender.com/ingest/sops

# Full ingestion
curl -X POST https://zendesk-rag-bot-pqbr.onrender.com/ingest/all
```

---

## How the Prompt Works

The LLM follows strict rules on every call:

1. **SOPs = source of truth** — always prioritised over everything
2. **Past tickets = reference only** — used for tone and pattern matching
3. **No hallucination** — if info isn't in SOPs, escalates to manual review
4. **Escalation** — returns `MANUAL_REVIEW_REQUIRED` when SOPs don't cover the issue
5. **Temperature 0.2** — deterministic, factual, consistent responses

---

## What Happens When SOP is Missing

If a ticket comes in that no SOP covers, the bot posts:
```
🤖 RAG Bot Suggestion

MANUAL_REVIEW_REQUIRED: This issue requires manual 
review by a support agent.
```
Confidence will show as `MANUAL_REVIEW` — use this as a signal to write a new SOP.

---

## Zendesk Webhook Setup

**Webhook:**
- URL: `https://zendesk-rag-bot-pqbr.onrender.com/webhook/zendesk`
- Method: `POST`
- Format: `JSON`

**Trigger condition:** Ticket is Created

**Trigger action body:**
```json
{
  "ticket_id": "{{ticket.id}}",
  "ticket_description": "{{ticket.description}}"
}
```

---

## Known Limitations & Next Steps

| Item | Status | Fix |
|---|---|---|
| Render sleeps after 15min | ⚠️ Free tier | Upgrade to Render Starter ($7/mo) |
| Ticket ingestion 401 error | ⚠️ Pending | Re-enable Zendesk API token access |
| Only 5 sample SOPs | ⚠️ | Add real company SOPs to sops.json |
| No daily re-ingestion | ⚠️ | Add Render cron job |

---

## Future Improvements

- Add more SOPs specific to your business
- Ingest resolved tickets for richer context
- Schedule daily re-ingestion via Render cron
- Add confidence threshold — only post if confidence is HIGH
- Track which SOPs are used most to identify gaps
