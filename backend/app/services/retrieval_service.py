# backend/app/services/retrieval_service.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.retrieval.search_client import SearchClient
from app.retrieval.vector import VectorSearchClient
from app.retrieval.hybrid import HybridSearchClient


class RetrievalService:
    """
    Unified retrieval service for BM25, vector, and hybrid search.

    Supported modes:
        - bm25
        - vector
        - hybrid
    """

    SUPPORTED_MODES = {"bm25", "vector", "hybrid"}

    def __init__(
        self,
        bm25_client: Optional[SearchClient] = None,
        vector_client: Optional[VectorSearchClient] = None,
        hybrid_client: Optional[HybridSearchClient] = None,
    ) -> None:
        self.bm25_client = bm25_client or SearchClient(
            host=settings.ELASTICSEARCH_HOST,
            index_name=settings.INDEX_NAME,
        )

        self.vector_client = vector_client or VectorSearchClient(
            host=getattr(settings, "QDRANT_HOST", "localhost"),
            port=getattr(settings, "QDRANT_PORT", 6333),
            collection_name=getattr(settings, "QDRANT_COLLECTION", "stackoverflow_vector"),
            model_name=getattr(settings, "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        )

        self.hybrid_client = hybrid_client or HybridSearchClient(
            bm25_client=self.bm25_client,
            vector_client=self.vector_client,
        )

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Run retrieval using the selected mode.

        Args:
            query: User query string.
            mode: Retrieval mode: "bm25", "vector", or "hybrid".
            top_k: Number of top results to return.

        Returns:
            List of retrieved documents.

        Raises:
            ValueError: If query is empty, mode is invalid, or top_k is invalid.
            RuntimeError: If underlying retrieval fails.
        """
        query = self._validate_query(query)
        mode = self._normalize_mode(mode)
        top_k = self._validate_top_k(top_k)

        try:
            if mode == "bm25":
                results = self.bm25_client.search(query=query, top_k=top_k)
            elif mode == "vector":
                results = self.vector_client.search(query=query, top_k=top_k)
            elif mode == "hybrid":
                results = self.hybrid_client.search(query=query, top_k=top_k)
            else:
                # This should never happen because mode is validated above.
                raise ValueError(f"Unsupported retrieval mode: {mode}")

            return results if results is not None else []

        except Exception as e:
            raise RuntimeError(
                f"Retrieval failed for mode='{mode}', query='{query}', top_k={top_k}: {e}"
            ) from e

    def get_supported_modes(self) -> List[str]:
        """Return supported retrieval modes."""
        return sorted(self.SUPPORTED_MODES)

    def healthcheck(self) -> Dict[str, Any]:
        """
        Lightweight service-level health info.

        This does not guarantee that every backend dependency is reachable,
        but it helps to inspect initialization state.
        """
        return {
            "status": "ok",
            "supported_modes": self.get_supported_modes(),
            "clients": {
                "bm25_client": self.bm25_client.__class__.__name__,
                "vector_client": self.vector_client.__class__.__name__,
                "hybrid_client": self.hybrid_client.__class__.__name__,
            },
            "config": {
                "elasticsearch_host": getattr(settings, "ELASTICSEARCH_HOST", None),
                "index_name": getattr(settings, "INDEX_NAME", None),
                "qdrant_host": getattr(settings, "QDRANT_HOST", None),
                "qdrant_port": getattr(settings, "QDRANT_PORT", None),
                "qdrant_collection": getattr(settings, "QDRANT_COLLECTION", None),
                "embedding_model_name": getattr(settings, "EMBEDDING_MODEL_NAME", None),
            },
        }

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        return query

    @classmethod
    def _normalize_mode(cls, mode: str) -> str:
        if not isinstance(mode, str):
            raise ValueError("Mode must be a string.")

        mode = mode.strip().lower()
        if mode not in cls.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported retrieval mode: '{mode}'. "
                f"Supported modes: {sorted(cls.SUPPORTED_MODES)}"
            )

        return mode

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int):
            raise ValueError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if top_k > 100:
            raise ValueError("top_k must be <= 100.")

        return top_k