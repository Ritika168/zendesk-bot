"""
Text preprocessing — clean HTML, chunk text, extract ticket summaries.
"""

import re
import html
from typing import Iterator
from html.parser import HTMLParser

from app.config import settings


class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper (no extra deps needed)."""
    def __init__(self):
        super().__init__()
        self.reset()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    stripper = _HTMLStripper()
    stripper.feed(html.unescape(text))
    return stripper.get_text()


def clean_text(text: str) -> str:
    """Normalise whitespace and remove non-printable chars."""
    text = strip_html(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = settings.MAX_CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> list[str]:
    """
    Word-level sliding window chunker.
    chunk_size / overlap are expressed in *words* (close enough to tokens
    for sentence-transformer models without needing a tokeniser).
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return [c for c in chunks if len(c.split()) > 10]  # drop tiny tail chunks


# ── SOP preprocessing ─────────────────────────────────────────────────────────

def preprocess_sop_article(article: dict) -> list[dict]:
    """
    Turn a Zendesk Help Center article into a list of chunk dicts.
    Each dict is ready for embedding + Pinecone upsert.
    """
    body = clean_text(article.get("body", ""))
    title = clean_text(article.get("title", ""))
    source_id = str(article.get("id", ""))

    if not body:
        return []

    # Prepend title to each chunk for context
    full_text = f"Title: {title}\n\n{body}"
    chunks = chunk_text(full_text)

    return [
        {
            "id": f"sop_{source_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "type": "SOP",
                "source_id": source_id,
                "title": title,
                "url": article.get("html_url", ""),
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(chunks)
    ]


# ── Ticket summary preprocessing ─────────────────────────────────────────────

def extract_ticket_summary(ticket: dict, comments: list[dict] | None = None) -> str:
    """
    Build a concise summary string for a resolved ticket.
    Uses subject + description + last public agent comment (resolution).
    """
    subject = clean_text(ticket.get("subject", ""))
    description = clean_text(ticket.get("description", ""))

    resolution = ""
    if comments:
        # Last public comment from an agent = resolution
        agent_comments = [
            c for c in comments
            if not c.get("author", {}).get("role") == "end-user" and c.get("public")
        ]
        if agent_comments:
            resolution = clean_text(agent_comments[-1].get("body", ""))

    parts = [f"Issue: {subject}"]
    if description:
        # Truncate description to first 300 chars
        parts.append(f"Customer reported: {description[:300]}")
    if resolution:
        parts.append(f"Resolution: {resolution[:400]}")

    return "\n".join(parts)


def preprocess_ticket(ticket: dict, comments: list[dict] | None = None) -> list[dict]:
    """
    Turn a resolved ticket into a list of chunk dicts.
    Ticket summaries are generally short so usually 1 chunk.
    """
    summary = extract_ticket_summary(ticket, comments)
    if not summary or len(summary.split()) < 15:
        return []

    source_id = str(ticket.get("id", ""))
    chunks = chunk_text(summary)

    return [
        {
            "id": f"ticket_{source_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "type": "TICKET",
                "source_id": source_id,
                "title": clean_text(ticket.get("subject", f"Ticket {source_id}")),
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(chunks)
    ]
