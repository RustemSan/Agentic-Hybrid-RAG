from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI

from app.core.config import settings
from app.services.generator_service import GeneratorService
from app.services.retrieval_service import RetrievalService
from app.services.agent_router import RetrievalRouterAgent
from app.services.llm_client import LLMClient
from app.services.query_rewriter import QueryRewriterAgent

def _build_default_llm_client() -> LLMClient:
    """Создаём LLMClient из настроек, если не передан снаружи."""
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return LLMClient(client=openai_client, model=settings.OPENAI_MODEL)

class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generator_service: GeneratorService | None = None,
        llm_client: LLMClient | None = None,
        agent: RetrievalRouterAgent | None = None,
        rewriter: QueryRewriterAgent | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()
        self.generator_service = generator_service or GeneratorService()
        self.llm_client = llm_client or _build_default_llm_client()
        self.agent = agent or RetrievalRouterAgent(llm_client=self.llm_client)
        self.rewriter = rewriter or QueryRewriterAgent(llm_client=self.llm_client)


    def answer(self, query: str, mode: str = "hybrid", top_k: int | None = None, use_agent=False, use_rewriter: bool = False,) -> Dict[str, Any]:
        k = top_k or settings.GENERATION_TOP_K

        rewritten_query = query
        rewriter_used = False
        if use_rewriter:
            rewriter_used = True
            rewritten_query = self.rewriter.rewrite(query)


        if use_agent:
            selected_mode = self.agent.decide_mode(rewritten_query)
            agent_used = True
        else:
            selected_mode = mode
            agent_used = False

        retrieved_docs = self.retrieval_service.search(
            query=rewritten_query ,
            mode=selected_mode,
            top_k=k,
        )

        answer_text = self.generator_service.generate(
            query=query,
            documents=retrieved_docs,
        )

        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "mode": selected_mode,
            "agent_used": agent_used,
            "rewriter_used": rewriter_used,
            "retrieved_count": len(retrieved_docs),
            "answer": answer_text,
            "sources": retrieved_docs,
        }