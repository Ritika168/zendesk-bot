"""
Zendesk API client — wraps Help Center + Tickets endpoints.
Uses httpx async for all HTTP calls.
"""

import logging
from typing import AsyncIterator
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = f"https://{settings.ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"
HELP_CENTER_URL = f"https://{settings.ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/help_center"


class ZendeskClient:
    def __init__(self):
        auth = (f"{settings.ZENDESK_EMAIL}/token", settings.ZENDESK_API_TOKEN)
        self._client = httpx.AsyncClient(
            auth=auth,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

    # ── Help Center / SOPs ────────────────────────────────────────────────────

    async def fetch_sop_articles(self, limit: int = 200) -> list[dict]:
        """
        Fetch all published Help Center articles (SOPs).
        Handles Zendesk cursor-based pagination.
        """
        articles = []
        url = f"{HELP_CENTER_URL}/articles.json?per_page=100&sort_by=updated_at"

        while url and len(articles) < limit:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
            articles.extend(data.get("articles", []))
            url = data.get("next_page")
            logger.info(f"Fetched {len(articles)} SOP articles so far…")

        logger.info(f"Total SOP articles fetched: {len(articles)}")
        return articles[:limit]

    # ── Resolved Tickets ──────────────────────────────────────────────────────

    async def fetch_closed_tickets(self, limit: int = 500) -> list[dict]:
        """
        Fetch solved/closed tickets using the search API.
        Returns ticket objects with full description.
        """
        tickets = []
        url = (
            f"{BASE_URL}/search.json"
            f"?query=type:ticket status:solved&per_page=100&sort_by=updated_at&sort_order=desc"
        )

        while url and len(tickets) < limit:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
            tickets.extend(data.get("results", []))
            url = data.get("next_page")
            logger.info(f"Fetched {len(tickets)} closed tickets so far…")

        logger.info(f"Total closed tickets fetched: {len(tickets)}")
        return tickets[:limit]

    async def get_ticket_comments(self, ticket_id: int) -> list[dict]:
        """Fetch all comments for a ticket (used to find final resolution note)."""
        resp = await self._client.get(f"{BASE_URL}/tickets/{ticket_id}/comments.json")
        resp.raise_for_status()
        return resp.json().get("comments", [])

    # ── Post back to Zendesk ──────────────────────────────────────────────────

    async def post_internal_note(self, ticket_id: int, note: str) -> dict:
        """
        Add an internal note (visible only to agents) to a ticket.
        Used to surface the RAG-generated suggestion to agents.
        """
        payload = {
            "ticket": {
                "comment": {
                    "body": f"🤖 **RAG Bot Suggestion**\n\n{note}",
                    "public": False,  # internal note
                }
            }
        }
        resp = await self._client.put(
            f"{BASE_URL}/tickets/{ticket_id}.json",
            json=payload,
        )
        resp.raise_for_status()
        logger.info(f"Posted internal note to ticket {ticket_id}")
        return resp.json()

    async def aclose(self):
        await self._client.aclose()
