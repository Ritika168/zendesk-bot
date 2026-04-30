"""
LLM client — Groq API (free tier, LLaMA 3.1 8B Instant).
"""

import logging
from groq import AsyncGroq
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Zendesk support assistant. Your job is to suggest a helpful, accurate response to a customer ticket.

RULES (follow strictly):
1. SOPs are the PRIMARY source of truth. Always prioritise SOP instructions over anything else.
2. Past ticket summaries are REFERENCE ONLY.
3. Do NOT hallucinate steps or information not present in the provided context.
4. If the customer's message is vague or short (e.g. "please help", "not working"), use the ticket SUBJECT and any context clues to infer the issue and respond based on the most relevant SOP.
5. If the SOPs do not cover the issue at all, reply EXACTLY with: "MANUAL_REVIEW_REQUIRED: This issue requires manual review by a support agent."
6. Keep responses concise, professional, and actionable.
7. Use numbered steps when describing a procedure.
8. Do NOT mention that you are an AI or that you are using a knowledge base.
9. Address the customer directly using "you"/"your".
10. Always sign off as "Zendesk Support".
"""
SUMMARY_PROMPT = """You are a support knowledge base builder.
Given a resolved support ticket, generate a concise summary with exactly these sections:

PROBLEM: One sentence describing what the customer reported.
ACTIONS: Bullet points of key steps the agent took.
RESOLUTION: One sentence describing how it was resolved.
CATEGORY: One of: billing, authentication, technical, account, refund, other
TAGS: 3-5 keywords (comma separated)

Be factual. Only include information present in the ticket."""

RESPONSE_TEMPLATE = """## Customer Ticket
{ticket_description}

---

## Relevant SOPs (source of truth)
{sop_context}

---

## Similar Resolved Tickets (reference only)
{ticket_context}

---

Based on the ticket subject and description, identify the customer's core issue and draft a helpful support response using the SOPs above. If the description is vague, infer the issue from the subject line and respond accordingly."""


class LLMClient:
    def __init__(self):
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    async def generate(
        self,
        ticket_description: str,
        sop_chunks: list[dict],
        ticket_chunks: list[dict],
    ) -> tuple[str, str]:
        sop_context = self._format_chunks(sop_chunks) or "No relevant SOPs found."
        ticket_context = self._format_chunks(ticket_chunks) or "No similar past tickets found."

        user_content = RESPONSE_TEMPLATE.format(
            ticket_description=ticket_description,
            sop_context=sop_context,
            ticket_context=ticket_context,
        )

        logger.info(f"Sending prompt to Groq ({settings.GROQ_MODEL})...")
        response = await self._client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=600,
        )

        text = response.choices[0].message.content.strip()
        confidence = self._assess_confidence(text, sop_chunks)
        return text, confidence

    @staticmethod
    def _format_chunks(chunks: list[dict]) -> str:
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{c['type']} {i}] {c['title']}\n{c['text']}")
        return "\n\n".join(parts)

    @staticmethod
    def _assess_confidence(response_text: str, sop_chunks: list[dict]) -> str:
        if "MANUAL_REVIEW_REQUIRED" in response_text:
            return "MANUAL_REVIEW"
        if not sop_chunks:
            return "MEDIUM"
        top_score = max((c.get("score", 0) for c in sop_chunks), default=0)
        return "HIGH" if top_score >= 0.75 else "MEDIUM"
