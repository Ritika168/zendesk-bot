# Zendesk RAG Bot

> A production-ready, cloud-deployed RAG chatbot that generates intelligent support responses using your Zendesk SOPs and resolved ticket history.

**Live URL:** `https://zendesk-rag-bot-pqbr.onrender.com`

---

## Current Status

| Component | Status |
|---|---|
| FastAPI backend | ✅ Live on Render |
| Pinecone vector DB | ✅ Connected (384-dim, cosine) |
| HuggingFace embeddings | ✅ Working (all-MiniLM-L6-v2) |
| Groq LLM | ✅ Working (llama-3.1-8b-instant) |
| SOPs loaded | ✅ 5 SOPs in Pinecone |
| Zendesk tickets | ⏳ Pending (fix API token) |
| Zendesk webhook | ⏳ Not connected yet |

---

## Architecture

```
INGESTION PIPELINE
──────────────────
sops.json (local)          Zendesk Closed Tickets
      │                           │
      ▼                           ▼
   Preprocess              Preprocess
  (clean + chunk)         (extract summary)
      │                           │
      ▼                           ▼
HuggingFace Embeddings API (all-MiniLM-L6-v2, 384-dim, free)
      │
      ▼
Pinecone Index (type: SOP | TICKET)

QUERY PIPELINE
──────────────
New Ticket Description
      │
      ▼
HuggingFace Embed query
      │
      ▼
Pinecone Query ──► Top 3 SOPs + Top 3 Tickets
      │
      ▼
Groq LLM (llama-3.1-8b-instant, free tier)
      │
      ▼
Response JSON  ──► (optional) Zendesk Internal Note
```

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| Backend | FastAPI + Uvicorn | Free |
| Embeddings | HuggingFace Inference API | Free |
| Vector DB | Pinecone (serverless) | Free tier |
| LLM | Groq (llama-3.1-8b-instant) | Free tier |
| Deployment | Render.com | Free tier |
| SOPs | Local sops.json | Free |

**Total cost: $0/month**

---

## Project Structure

```
zendesk-rag-bot/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app + all endpoints
│   ├── config.py         # Environment variable settings
│   ├── models.py         # Pydantic request/response models
│   ├── rag_pipeline.py   # Core orchestrator
│   ├── zendesk_client.py # Zendesk API client
│   ├── preprocessor.py   # Text cleaning + chunking
│   ├── embeddings.py     # HuggingFace embedding service
│   ├── vector_store.py   # Pinecone upsert + query
│   └── llm_client.py     # Groq LLM client + prompt
├── scripts/
│   └── ingest.py         # Standalone ingestion CLI
├── tests/
│   └── test_preprocessor.py
├── sops.json             # Your SOP knowledge base
├── .env.example
├── render.yaml
└── requirements.txt
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API docs (Swagger UI) |
| `POST` | `/ingest/sops` | Load SOPs into Pinecone |
| `POST` | `/ingest/tickets` | Load resolved tickets into Pinecone |
| `POST` | `/ingest/all` | Full ingestion |
| `POST` | `/query` | Generate response for a ticket |
| `POST` | `/webhook/zendesk` | Zendesk webhook receiver |

---

## Quick Test

```bash
# Health check
curl https://zendesk-rag-bot-pqbr.onrender.com/health

# Query the bot
curl -X POST https://zendesk-rag-bot-pqbr.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"ticket_description": "I forgot my password and cannot log in"}'

# Re-run ingestion (after updating sops.json)
curl -X POST https://zendesk-rag-bot-pqbr.onrender.com/ingest/sops
```

---

## Adding Your Own SOPs

Edit `sops.json` in the root of the repo. Add as many as you need:

```json
[
  {
    "id": "sop-006",
    "title": "Your SOP Title",
    "body": "Step 1: Do this.\nStep 2: Do that.\nStep 3: Escalate if needed."
  }
]
```

After saving and pushing to GitHub, re-run ingestion:
```bash
curl -X POST https://zendesk-rag-bot-pqbr.onrender.com/ingest/sops
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ZENDESK_SUBDOMAIN` | ✅ | e.g. `demo-87927` |
| `ZENDESK_EMAIL` | ✅ | Agent login email |
| `ZENDESK_API_TOKEN` | ✅ | Zendesk API token |
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `PINECONE_INDEX_NAME` | ✅ | `zendesk-rag` |
| `GROQ_API_KEY` | ✅ | Groq API key |
| `GROQ_MODEL` | ✅ | `llama-3.1-8b-instant` |
| `HF_API_TOKEN` | ✅ | HuggingFace token |
| `AUTO_POST_TO_ZENDESK` | ⬜ | `false` (set `true` after webhook setup) |
| `TOP_K_RESULTS` | ⬜ | Default: `3` |

---

## Connecting Zendesk Webhook (TODO)

When ready, do this to make the bot auto-respond to new tickets:

1. Zendesk Admin → **Objects & Rules → Triggers → Add Trigger**
2. Condition: **Ticket is Created**
3. Action: **Notify active webhook**
4. Webhook URL: `https://zendesk-rag-bot-pqbr.onrender.com/webhook/zendesk`
5. JSON body:
```json
{
  "ticket_id": "{{ticket.id}}",
  "ticket_description": "{{ticket.description}}"
}
```
6. Set `AUTO_POST_TO_ZENDESK=true` in Render environment

---

## Prompt Rules

The LLM follows these rules on every call:
1. SOPs are the **source of truth** — always prioritised
2. Past tickets are **reference only**
3. No hallucination — if info isn't in context, escalates to manual review
4. Returns `MANUAL_REVIEW_REQUIRED` if SOPs don't cover the issue
5. Temperature: 0.2 (deterministic, factual)

---

## Note on Free Tier

Render free tier **sleeps after 15 min of inactivity**. First request after sleep takes ~50 seconds to wake up. Upgrade to Render Starter ($7/mo) for always-on.
