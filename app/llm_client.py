"""
LLM client — Groq API (free tier, LLaMA 3 8B).
Groq offers a generous free tier: ~14,400 requests/day at low latency.
Fallback: switch GROQ_MODEL to "mixtral-8x7b-32768" for longer context.
"""

import logging
from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────
# Rules are embedded directly in the system prompt so the LLM always sees them.

SYSTEM_PROMPT = """You are a Zendesk support assistant. Your job is to suggest a helpful, accurate response to a customer ticket.

RULES (follow strictly):
1. SOPs are the PRIMARY source of truth. Always prioritise SOP instructions over anything else.
2. Past ticket summaries are REFERENCE ONLY — use them to understand how similar issues were handled.
3. Do NOT hallucinate steps, procedures, or information that is not present in the provided context.
4. If the SOPs do not cover the issue adequately, reply EXACTLY with:
   "MANUAL_REVIEW_REQUIRED: This issue requires manual review by a support agent."
5. Keep your response concise, professional, and actionable.
6. Use numbered steps when describing a procedure.
7. Do NOT mention that you are an AI or that you are using a knowledge base.
8. Address the customer directly (use "you"/"your").
"""

RESPONSE_TEMPLATE = """## Customer Issue
{ticket_description}

---

## Relevant SOPs (source of truth)
{sop_context}

---

## Similar Resolved Tickets (reference only)
{ticket_context}

---

Based on the SOPs above, draft a support response to the customer issue. If the SOPs do not contain enough information, output MANUAL_REVIEW_REQUIRED."""


class LLMClient:
    def __init__(self):
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    async def generate(
        self,
        ticket_description: str,
        sop_chunks: list[dict],
        ticket_chunks: list[dict],
    ) -> tuple[str, str]:
        """
        Generate a response using Groq.
        Returns (response_text, confidence_level).
        confidence: "HIGH" | "MEDIUM" | "MANUAL_REVIEW"
        """
        sop_context = self._format_chunks(sop_chunks, "SOP") or "No relevant SOPs found."
        ticket_context = self._format_chunks(ticket_chunks, "TICKET") or "No similar past tickets found."

        user_content = RESPONSE_TEMPLATE.format(
            ticket_description=ticket_description,
            sop_context=sop_context,
            ticket_context=ticket_context,
        )

        logger.info(f"Sending prompt to Groq ({settings.GROQ_MODEL})…")
        response = await self._client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,    # Low temp → deterministic, factual
            max_tokens=600,
        )

        text = response.choices[0].message.content.strip()
        confidence = self._assess_confidence(text, sop_chunks)
        return text, confidence

    @staticmethod
    def _format_chunks(chunks: list[dict], label: str) -> str:
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{label} {i}] {c['title']}\n{c['text']}")
        return "\n\n".join(parts)

    @staticmethod
    def _assess_confidence(response_text: str, sop_chunks: list[dict]) -> str:
        if "MANUAL_REVIEW_REQUIRED" in response_text:
            return "MANUAL_REVIEW"
        if not sop_chunks:
            return "MEDIUM"
        # High confidence when strong SOP matches exist (score > 0.75)
        top_score = max((c.get("score", 0) for c in sop_chunks), default=0)
        return "HIGH" if top_score >= 0.75 else "MEDIUM"
