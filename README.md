# Zendesk RAG Bot

> A production-ready, cloud-deployable RAG (Retrieval-Augmented Generation) chatbot that generates intelligent support responses using your Zendesk SOPs and resolved ticket history.

---

## Architecture Overview

```
New Ticket (Zendesk Webhook)
        │
        ▼
┌──────────────────┐
│   FastAPI App    │  ← /webhook/zendesk
│   (Render.com)   │
└────────┬─────────┘
         │
         ▼
  Embed ticket description
  (all-MiniLM-L6-v2, local)
         │
         ▼
┌────────────────────────────┐
│       Pinecone Query       │
│  ┌──────────┬───────────┐  │
│  │ SOP      │ TICKET    │  │
│  │ chunks   │ summaries │  │
│  └──────────┴───────────┘  │
└────────────┬───────────────┘
             │
             ▼
   Build prompt (SOP = source of truth,
   tickets = reference only)
             │
             ▼
┌────────────────────┐
│   Groq LLM API     │
│  (LLaMA 3, free)   │
└────────┬───────────┘
         │
         ▼
  Post internal note → Zendesk
```

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| Backend | FastAPI + Uvicorn | Free |
| Embeddings | sentence-transformers (local) | Free |
| Vector DB | Pinecone | Free tier |
| LLM | Groq (LLaMA 3 8B) | Free tier |
| Deployment | Render.com | Free tier |
| Source data | Zendesk Help Center + Tickets API | Existing |

**Total infra cost: $0/month** (within free tier limits)

---

## Project Structure

```
zendesk-rag-bot/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, endpoints, webhook handler
│   ├── config.py         # Pydantic settings (env var loading)
│   ├── models.py         # Request / response Pydantic models
│   ├── rag_pipeline.py   # Orchestrator: ingest + query flow
│   ├── zendesk_client.py # Zendesk API (articles + tickets + posting)
│   ├── preprocessor.py   # HTML cleaning, chunking, summary extraction
│   ├── embeddings.py     # sentence-transformers wrapper
│   ├── vector_store.py   # Pinecone upsert + query
│   └── llm_client.py     # Groq LLM client + prompt template
├── scripts/
│   └── ingest.py         # Standalone ingestion CLI
├── tests/
│   └── test_preprocessor.py
├── .env.example
├── .gitignore
├── render.yaml           # One-click Render deployment
└── requirements.txt
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/ingest/sops` | Fetch + embed Zendesk SOP articles |
| `POST` | `/ingest/tickets` | Fetch + embed resolved ticket summaries |
| `POST` | `/ingest/all` | Full ingestion (SOPs + tickets) |
| `POST` | `/query` | Generate response for a ticket description |
| `POST` | `/webhook/zendesk` | Zendesk webhook receiver |

### POST /query — Example

**Request:**
```json
{
  "ticket_description": "I forgot my password and can't log into my account. The reset email isn't arriving.",
  "ticket_id": "optional-123"
}
```

**Response:**
```json
{
  "ticket_id": "optional-123",
  "response": "Thank you for reaching out. To resolve your login issue:\n\n1. Check your spam/junk folder for the reset email.\n2. Ensure you're using the email address registered to your account.\n3. Request a new reset link at https://example.com/reset — links expire after 30 minutes.\n4. If the issue persists, contact us with your account username.",
  "retrieved_sops": [...],
  "retrieved_tickets": [...],
  "confidence": "HIGH"
}
```

---

## Pinecone Schema

```
Index name : zendesk-rag
Dimension  : 384 (all-MiniLM-L6-v2)
Metric     : cosine
Spec       : serverless, aws us-east-1 (free tier)

Vector metadata fields:
  type        : "SOP" | "TICKET"
  source_id   : Zendesk article ID or ticket ID
  title       : Article title or ticket subject
  chunk_index : Position within the source document
  text        : Raw chunk text (stored for retrieval)
  url         : Article URL (SOP only)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourorg/zendesk-rag-bot.git
cd zendesk-rag-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

Get your free API keys:
- **Pinecone**: https://app.pinecone.io (free tier, no credit card)
- **Groq**: https://console.groq.com (14,400 free requests/day)
- **Zendesk**: Admin → Apps & Integrations → Zendesk API

### 3. Run ingestion (one-time setup)

```bash
python scripts/ingest.py --mode all
```

### 4. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test it

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"ticket_description": "I cannot reset my password, the email never arrives."}'
```

---

## Deployment to Render (Free)

1. Push code to GitHub
2. Go to https://dashboard.render.com → **New → Blueprint**
3. Connect your GitHub repo
4. Render reads `render.yaml` automatically
5. Set environment variables in the Render dashboard
6. Deploy! Your bot will be live at `https://your-app.onrender.com`

### Set up Zendesk Webhook Trigger

1. Zendesk Admin → Objects & Rules → Triggers → Add Trigger
2. Condition: **Ticket is Created**
3. Action: **Notify webhook**
4. Webhook URL: `https://your-app.onrender.com/webhook/zendesk`
5. Method: `POST`
6. Body:
```json
{
  "ticket_id": "{{ticket.id}}",
  "ticket_description": "{{ticket.description}}"
}
```

### Schedule daily re-ingestion (Render Cron Job)

In `render.yaml`, add:
```yaml
  - type: cron
    name: daily-ingest
    env: python
    schedule: "0 2 * * *"
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/ingest.py --mode all
```

---

## Prompt Rules (LLM Behaviour)

The system prompt enforces these rules on every LLM call:

1. **SOPs = source of truth** — always prioritised
2. **Ticket summaries = reference only** — used for tone/pattern matching
3. **No hallucination** — if information isn't in the context, the bot says so
4. **Manual review escalation** — if SOPs don't cover the issue, returns `MANUAL_REVIEW_REQUIRED`
5. **Concise + actionable** — numbered steps, no waffle
6. **Low temperature (0.2)** — deterministic, factual output

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `ZENDESK_SUBDOMAIN` | ✅ | Your Zendesk subdomain |
| `ZENDESK_EMAIL` | ✅ | Agent email for API auth |
| `ZENDESK_API_TOKEN` | ✅ | Zendesk API token |
| `ZENDESK_WEBHOOK_SECRET` | ⬜ | HMAC secret for webhook verification |
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `PINECONE_INDEX_NAME` | ⬜ | Default: `zendesk-rag` |
| `GROQ_API_KEY` | ✅ | Groq API key |
| `GROQ_MODEL` | ⬜ | Default: `llama3-8b-8192` |
| `AUTO_POST_TO_ZENDESK` | ⬜ | Auto-post suggestions (default: false) |
| `TOP_K_RESULTS` | ⬜ | Pinecone top-k per source (default: 3) |
| `MAX_CHUNK_SIZE` | ⬜ | Words per chunk (default: 500) |
| `CHUNK_OVERLAP` | ⬜ | Overlap words (default: 50) |

---

## Cost Estimate

| Service | Free Tier | When you'd upgrade |
|---|---|---|
| Render.com | 750 hrs/month (sleeps after 15min idle) | $7/mo for always-on |
| Pinecone | 1 index, 100k vectors | Paid if > 100k chunks |
| Groq | 14,400 req/day, 6000 tokens/min | Very high volume |
| sentence-transformers | Free (local inference) | Never |

**For most support teams: $0/month.**
