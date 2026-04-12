from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic-Hybrid-RAG"
    API_V1_STR: str = "/api/v1"

    # =========================
    # Elasticsearch settings
    # =========================
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "http://127.0.0.1:9200")
    INDEX_NAME: str = os.getenv("INDEX_NAME", "stackoverflow_bm25")

    # =========================
    # Qdrant settings
    # =========================
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "stackoverflow_vector")

    # =========================
    # Embedding model settings
    # =========================
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-small-en-v1.5"
    )

    # =========================
    # Retrieval defaults
    # =========================
    DEFAULT_RETRIEVAL_MODE: str = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid")
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", 5))
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", 20))

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