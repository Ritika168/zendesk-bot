from pydantic import BaseModel, Field, validator
from typing import Optional


class IngestRequest(BaseModel):
    limit: int = Field(default=200, ge=1, le=2000)


class QueryRequest(BaseModel):
    ticket_description: str = Field(..., min_length=10)
    ticket_id: Optional[str] = None


class TicketWebhookPayload(BaseModel):
    ticket_id: str  # Zendesk sends as string
    ticket_description: str

    @property
    def ticket_id_int(self) -> int:
        return int(self.ticket_id)


class RetrievedChunk(BaseModel):
    id: str
    score: float
    type: str
    source_id: str
    title: str
    text: str


class RAGResponse(BaseModel):
    ticket_id: Optional[str]
    response: str
    retrieved_sops: list[RetrievedChunk]
    retrieved_tickets: list[RetrievedChunk]
    confidence: str
