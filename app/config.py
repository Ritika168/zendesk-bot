from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    ZENDESK_SUBDOMAIN: str
    ZENDESK_EMAIL: str
    ZENDESK_API_TOKEN: str
    ZENDESK_WEBHOOK_SECRET: str = ""
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "zendesk-rag"
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-8b-8192"
    EMBEDDING_DIM: int = 384   # all-MiniLM-L6-v2 = 384 dims
    AUTO_POST_TO_ZENDESK: bool = False
    TOP_K_RESULTS: int = 3
    MAX_CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
