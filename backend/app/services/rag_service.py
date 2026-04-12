from __future__ import annotations

from typing import Any, Dict

from app.core.config import settings
from app.services.generator_service import GeneratorService
from app.services.retrieval_service import RetrievalService


class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generator_service: GeneratorService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()
        self.generator_service = generator_service or GeneratorService()

    def answer(self, query: str, mode: str = "hybrid", top_k: int | None = None) -> Dict[str, Any]:
        k = top_k or settings.GENERATION_TOP_K

        retrieved_docs = self.retrieval_service.search(
            query=query,
            mode=mode,
            top_k=k,
        )

        answer_text = self.generator_service.generate(
            query=query,
            documents=retrieved_docs,
        )

        return {
            "query": query,
            "mode": mode,
            "retrieved_count": len(retrieved_docs),
            "answer": answer_text,
            "sources": retrieved_docs,
        }