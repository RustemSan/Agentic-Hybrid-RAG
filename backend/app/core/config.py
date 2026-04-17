from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic-Hybrid-RAG"
    API_V1_STR: str = "/api/v1"

    # =========================
    # Elasticsearch settings
    # =========================
    ELASTICSEARCH_HOST: str = "http://127.0.0.1:9200"
    INDEX_NAME: str = "stackoverflow_bm25"

    # =========================
    # Qdrant settings
    # =========================
    QDRANT_HOST: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "stackoverflow_vector"

    # =========================
    # Embedding model settings
    # =========================
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

    # =========================
    # Retrieval defaults
    # =========================
    DEFAULT_RETRIEVAL_MODE: str = "hybrid"
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20

    # =========================
    # LLM / Generation settings
    # =========================

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-5.4"
    GENERATION_TOP_K: int = 4
    GENERATION_MAX_OUTPUT_TOKENS: int = 400
    GENERATION_TEMPERATURE: float = 0.2

    # =========================
    # CORS settings
    # =========================
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()