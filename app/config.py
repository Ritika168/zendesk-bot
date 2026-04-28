"""
Configuration — loaded from environment variables.
Copy .env.example to .env and fill in your values.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Zendesk ───────────────────────────────────────────────────────────────
    ZENDESK_SUBDOMAIN: str           # e.g. "mycompany" → mycompany.zendesk.com
    ZENDESK_EMAIL: str               # Agent email used for API auth
    ZENDESK_API_TOKEN: str           # Zendesk API token
    ZENDESK_WEBHOOK_SECRET: str = "" # Optional: for verifying webhook signatures

    # ── Pinecone ──────────────────────────────────────────────────────────────
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "zendesk-rag"
    PINECONE_ENVIRONMENT: str = "gcp-starter"  # Free tier environment

    # ── LLM (Groq — free tier, fast inference) ────────────────────────────────
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-8b-8192"          # Free, fast, capable

    # nomic-embed-text-v1_5 = 768-dim (via Groq, free, no local RAM)
    EMBEDDING_DIM: int = 768
    

    # ── App behaviour ─────────────────────────────────────────────────────────
    AUTO_POST_TO_ZENDESK: bool = False           # Set True to auto-post notes
    TOP_K_RESULTS: int = 3                       # Pinecone top-k per source
    MAX_CHUNK_SIZE: int = 500                    # tokens per chunk
    CHUNK_OVERLAP: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
