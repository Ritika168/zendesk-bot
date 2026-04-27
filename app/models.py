"""
Pydantic models for API request / response payloads.
"""

from pydantic import BaseModel, Field
from typing import Optional


class IngestRequest(BaseModel):
    limit: int = Field(default=200, ge=1, le=2000)


class QueryRequest(BaseModel):
    ticket_description: str = Field(..., min_length=10)
    ticket_id: Optional[str] = None


class TicketWebhookPayload(BaseModel):
    """
    Expected shape of Zendesk webhook POST body.
    Configure your Zendesk webhook trigger to send this JSON:

    {
      "ticket_id": "{{ticket.id}}",
      "ticket_description": "{{ticket.description}}"
    }
    """
    ticket_id: int
    ticket_description: str


class RetrievedChunk(BaseModel):
    id: str
    score: float
    type: str          # "SOP" or "TICKET"
    source_id: str
    title: str
    text: str


class RAGResponse(BaseModel):
    ticket_id: Optional[str]
    response: str
    retrieved_sops: list[RetrievedChunk]
    retrieved_tickets: list[RetrievedChunk]
    confidence: str    # "HIGH" | "MEDIUM" | "MANUAL_REVIEW"
