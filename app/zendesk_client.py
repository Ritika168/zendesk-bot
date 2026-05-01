"""
Zendesk API client — wraps Help Center + Tickets endpoints.
"""

import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = f"https://{settings.ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"


class ZendeskClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            auth=(f"{settings.ZENDESK_EMAIL}/token", settings.ZENDESK_API_TOKEN),
            headers={"Content-Type": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )

    # ── Help Center / SOPs ────────────────────────────────────────────────────

    async def fetch_sop_articles(self, limit: int = 200) -> list[dict]:
        """Fetch published Help Center articles. Returns empty if Guide not enabled."""
        articles = []
        url = f"{BASE_URL}/help_center/articles.json?per_page=100&sort_by=updated_at"

        while url and len(articles) < limit:
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
                data = resp.json()
                articles.extend(data.get("articles", []))
                url = data.get("next_page")
                logger.info(f"Fetched {len(articles)} SOP articles so far...")
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (302, 401, 403, 404):
                    logger.warning("Help Center not accessible. Skipping SOP ingestion from Zendesk.")
                    break
                raise

        logger.info(f"Total SOP articles fetched: {len(articles)}")
        return articles[:limit]

    # ── Resolved Tickets ──────────────────────────────────────────────────────

    async def fetch_closed_tickets(self, limit: int = 500) -> list[dict]:
        """Fetch solved/closed tickets using the search API."""
        tickets = []
        url = (
            f"{BASE_URL}/search.json"
            f"?query=type:ticket status:solved"
            f"&per_page=100&sort_by=updated_at&sort_order=desc"
        )

        while url and len(tickets) < limit:
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
                data = resp.json()
                tickets.extend(data.get("results", []))
                url = data.get("next_page")
                logger.info(f"Fetched {len(tickets)} closed tickets so far...")
            except httpx.HTTPStatusError as e:
                logger.warning(f"Could not fetch closed tickets: {e}")
                break

        logger.info(f"Total closed tickets fetched: {len(tickets)}")
        return tickets[:limit]

    async def get_ticket_comments(self, ticket_id: int) -> list[dict]:
        """Fetch all comments for a ticket."""
        resp = await self._client.get(
            f"{BASE_URL}/tickets/{ticket_id}/comments.json"
        )
        resp.raise_for_status()
        return resp.json().get("comments", [])

    # ── Post back to Zendesk ──────────────────────────────────────────────────

    async def post_internal_note(self, ticket_id: int, note: str) -> dict:
        """Add an internal note (visible only to agents) to a ticket."""
        payload = {
            "ticket": {
                "comment": {
                    "body": note,
                    "public": False,
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
