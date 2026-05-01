"""
LLM client — Groq API (free tier, LLaMA 3.1 8B Instant).

Two main functions:
1. generate()        — respond to a new incoming ticket using SOPs + past tickets
2. summarise_ticket() — summarise a closed ticket into structured knowledge
"""

import logging
from groq import AsyncGroq
from app.config import settings

logger = logging.getLogger(__name__)


# ── Prompt 1: Respond to new ticket ──────────────────────────────────────────

SYSTEM_PROMPT = """You are a Zendesk support assistant. Your job is to suggest a helpful, accurate response to a customer ticket.

RULES (follow strictly):
1. SOPs are the PRIMARY source of truth. Always prioritise SOP instructions over anything else.
2. Past ticket summaries are REFERENCE ONLY — use them to understand how similar issues were handled.
3. Do NOT hallucinate steps or information not present in the provided context.
4. If the customer message is vague or short (e.g. "please help"), use the ticket SUBJECT to infer the issue and respond based on the most relevant SOP.
5. If the SOPs do not cover the issue at all, reply EXACTLY with: "MANUAL_REVIEW_REQUIRED: This issue requires manual review by a support agent."
6. Keep responses concise, professional, and actionable.
7. Use numbered steps when describing a procedure.
8. Do NOT mention that you are an AI or that you are using a knowledge base.
9. Address the customer directly using "you" and "your".
10. Always sign off as "Zendesk Support".
"""

RESPONSE_TEMPLATE = """## Customer Ticket
{ticket_description}

---

## Relevant SOPs (source of truth — follow these exactly)
{sop_context}

---

## Similar Resolved Tickets (reference only — shows how similar issues were handled)
{ticket_context}

---

Based on the ticket subject and description, identify the customer's core issue.
Draft a helpful, professional support response using the SOPs above.
If the description is vague, infer the issue from the subject line and respond accordingly.
If no SOP covers this issue, output MANUAL_REVIEW_REQUIRED."""


# ── Prompt 2: Summarise a closed ticket ──────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """You are a support knowledge base builder.
Your job is to read a resolved support ticket and produce a clean, structured summary.
This summary will be stored and used to help answer similar tickets in the future.
Be factual. Only include information actually present in the ticket — never invent details."""

SUMMARY_TEMPLATE = """Here is a resolved support ticket:

SUBJECT: {subject}

CUSTOMER MESSAGE:
{description}

AGENT COMMENTS AND ACTIONS:
{comments}

---

Generate a structured summary with EXACTLY these sections and labels:

PROBLEM: (one sentence — what did the customer report?)
ACTIONS: (bullet points — what steps did the agent take?)
RESOLUTION: (one sentence — how was it finally resolved?)
CATEGORY: (pick exactly one: billing / authentication / technical / account / refund / other)
TAGS: (3 to 5 keywords, comma separated, lowercase)

Example format:
PROBLEM: Customer could not log in due to forgotten password.
ACTIONS:
- Asked customer to use Forgot Password link
- Verified registered email address
- Resent password reset email
RESOLUTION: Customer successfully reset password using the emailed link.
CATEGORY: authentication
TAGS: password, login, reset, email, access"""


# ── Prompt 3: Classify incoming ticket category ───────────────────────────────

CLASSIFY_TEMPLATE = """Read this support ticket and reply with exactly ONE category word.

Ticket: {ticket_description}

Choose from: billing, authentication, technical, account, refund, other

Reply with only the single category word, nothing else."""


class LLMClient:
    def __init__(self):
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    # ── Generate response for new ticket ─────────────────────────────────────

    async def generate(
        self,
        ticket_description: str,
        sop_chunks: list[dict],
        ticket_chunks: list[dict],
    ) -> tuple[str, str]:
        """
        Generate a suggested reply for an incoming ticket.
        Returns (response_text, confidence_level).
        confidence: "HIGH" | "MEDIUM" | "MANUAL_REVIEW"
        """
        sop_context = self._format_chunks(sop_chunks) or "No relevant SOPs found."
        ticket_context = self._format_chunks(ticket_chunks) or "No similar past tickets found."

        user_content = RESPONSE_TEMPLATE.format(
            ticket_description=ticket_description,
            sop_context=sop_context,
            ticket_context=ticket_context,
        )

        logger.info(f"Sending response prompt to Groq ({settings.GROQ_MODEL})...")
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
        logger.info(f"Response generated. Confidence: {confidence}")
        return text, confidence

    # ── Summarise a closed ticket ─────────────────────────────────────────────

    async def summarise_ticket(
        self,
        subject: str,
        description: str,
        comments: list[dict],
    ) -> dict:
        """
        Summarise a resolved ticket into structured knowledge.
        Returns a dict with: problem, actions, resolution, category, tags, full_summary
        """
        # Format agent comments — skip system/automated messages
        comment_text = ""
        for c in comments:
            author_role = c.get("author", {}).get("role", "unknown")
            body = c.get("body", "").strip()
            if body and len(body) > 10:
                comment_text += f"[{author_role}]: {body[:500]}\n\n"

        if not comment_text:
            comment_text = "No agent comments recorded."

        user_content = SUMMARY_TEMPLATE.format(
            subject=subject,
            description=description[:800],
            comments=comment_text[:2000],
        )

        logger.info("Generating ticket summary...")
        response = await self._client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,   # Very low — we want factual, structured output
            max_tokens=400,
        )

        full_summary = response.choices[0].message.content.strip()
        logger.info(f"Summary generated:\n{full_summary}")

        # Parse the structured output into fields
        parsed = self._parse_summary(full_summary, subject)
        parsed["full_summary"] = full_summary
        return parsed

    # ── Classify ticket category ──────────────────────────────────────────────

    async def classify_ticket(self, ticket_description: str) -> str:
        """
        Classify the ticket into a category for filtered Pinecone retrieval.
        Returns one of: billing, authentication, technical, account, refund, other
        """
        try:
            response = await self._client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": CLASSIFY_TEMPLATE.format(
                            ticket_description=ticket_description[:500]
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=10,
            )
            category = response.choices[0].message.content.strip().lower()
            valid = {"billing", "authentication", "technical", "account", "refund", "other"}
            return category if category in valid else "other"
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return "other"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_chunks(chunks: list[dict]) -> str:
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{c['type']} {i}] {c['title']}\n{c['text']}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_summary(text: str, subject: str) -> dict:
        """Parse the structured LLM summary output into a dictionary."""
        result = {
            "problem": "",
            "actions": "",
            "resolution": "",
            "category": "other",
            "tags": "",
        }

        lines = text.split("\n")
        current_key = None
        buffer = []

        for line in lines:
            line = line.strip()
            if line.startswith("PROBLEM:"):
                if current_key and buffer:
                    result[current_key] = " ".join(buffer).strip()
                current_key = "problem"
                buffer = [line.replace("PROBLEM:", "").strip()]
            elif line.startswith("ACTIONS:"):
                if current_key and buffer:
                    result[current_key] = " ".join(buffer).strip()
                current_key = "actions"
                buffer = [line.replace("ACTIONS:", "").strip()]
            elif line.startswith("RESOLUTION:"):
                if current_key and buffer:
                    result[current_key] = " ".join(buffer).strip()
                current_key = "resolution"
                buffer = [line.replace("RESOLUTION:", "").strip()]
            elif line.startswith("CATEGORY:"):
                if current_key and buffer:
                    result[current_key] = " ".join(buffer).strip()
                current_key = "category"
                cat = line.replace("CATEGORY:", "").strip().lower()
                valid = {"billing", "authentication", "technical", "account", "refund", "other"}
                result["category"] = cat if cat in valid else "other"
                current_key = None
                buffer = []
            elif line.startswith("TAGS:"):
                if current_key and buffer:
                    result[current_key] = " ".join(buffer).strip()
                result["tags"] = line.replace("TAGS:", "").strip().lower()
                current_key = None
                buffer = []
            elif current_key and line:
                buffer.append(line)

        if current_key and buffer:
            result[current_key] = " ".join(buffer).strip()

        # Fallback for problem if parsing failed
        if not result["problem"]:
            result["problem"] = subject

        return result

    @staticmethod
    def _assess_confidence(response_text: str, sop_chunks: list[dict]) -> str:
        if "MANUAL_REVIEW_REQUIRED" in response_text:
            return "MANUAL_REVIEW"
        if not sop_chunks:
            return "MEDIUM"
        top_score = max((c.get("score", 0) for c in sop_chunks), default=0)
        return "HIGH" if top_score >= 0.75 else "MEDIUM"
