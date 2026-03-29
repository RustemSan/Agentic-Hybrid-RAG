from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic-Hybrid-RAG"
    API_V1_STR: str = "/api/v1"

    #Elastic Settings
    ELASTICSEARCH_HOST: str = "http://127.0.0.1:9200"
    INDEX_NAME: str = "stackoverflow_bm25"

    #CORS settings
    # TODO!!
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

    class Config:
        case_sensitive = True

settings = Settings()