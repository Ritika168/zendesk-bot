#!/usr/bin/env python3
"""
scripts/ingest.py — standalone ingestion script.

Run this once to populate Pinecone, then optionally schedule it
(e.g., daily cron on Render) to keep the index fresh.

Usage:
    python scripts/ingest.py --mode all        # SOPs + tickets (default)
    python scripts/ingest.py --mode sops       # SOPs only
    python scripts/ingest.py --mode tickets    # Tickets only
    python scripts/ingest.py --mode all --sop-limit 500 --ticket-limit 1000
"""

import asyncio
import argparse
import logging
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def run(mode: str, sop_limit: int, ticket_limit: int):
    pipeline = RAGPipeline()

    if mode in ("all", "sops"):
        logger.info("=== Ingesting SOP articles ===")
        result = await pipeline.ingest_sops(limit=sop_limit)
        logger.info(f"SOPs done: {result}")

    if mode in ("all", "tickets"):
        logger.info("=== Ingesting resolved tickets ===")
        result = await pipeline.ingest_tickets(limit=ticket_limit)
        logger.info(f"Tickets done: {result}")

    # Print Pinecone index stats
    stats = pipeline.vector_store.stats()
    logger.info(f"Pinecone index stats: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Zendesk RAG ingestion script")
    parser.add_argument("--mode", choices=["all", "sops", "tickets"], default="all")
    parser.add_argument("--sop-limit", type=int, default=200)
    parser.add_argument("--ticket-limit", type=int, default=500)
    args = parser.parse_args()

    asyncio.run(run(args.mode, args.sop_limit, args.ticket_limit))


if __name__ == "__main__":
    main()
