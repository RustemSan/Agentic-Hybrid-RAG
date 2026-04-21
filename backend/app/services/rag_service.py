"""
rag_service.py — Central orchestrator for the RAG pipeline.

This service ties together all components:
  - RetrievalService  (BM25 / vector / hybrid search)
  - GeneratorService  (LLM answer generation)
  - LLMClient         (shared OpenAI client for agents)
  - RetrievalRouterAgent  (selects retrieval mode automatically)
  - QueryRewriterAgent    (expands query before retrieval)

The pipeline order when both agents are enabled:
  query → [Rewriter] → [Router] → Retrieval → [Generator] → answer

Each agent is optional and toggled per-request via use_agent / use_rewriter flags.
"""

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
    """
    Build a LLMClient from application settings.

    Called during RAGService initialization when no external llm_client
    is injected. Reads OPENAI_API_KEY and OPENAI_MODEL from config.
    """
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return LLMClient(client=openai_client, model=settings.OPENAI_MODEL)

class RAGService:
    """
        Orchestrates the full RAG pipeline: retrieval + optional agents + generation.
        """
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

        # llm_client is built once and shared by both agents below
        self.llm_client = llm_client or _build_default_llm_client()
        self.agent = agent or RetrievalRouterAgent(llm_client=self.llm_client)
        self.rewriter = rewriter or QueryRewriterAgent(llm_client=self.llm_client)


    def answer(self, query: str, mode: str = "hybrid", top_k: int | None = None, use_agent=False, use_rewriter: bool = False,) -> Dict[str, Any]:
        k = top_k or settings.GENERATION_TOP_K

        # --- Step 1: Query Rewriting ---
        rewritten_query = query
        rewriter_used = False
        if use_rewriter:
            rewriter_used = True
            rewritten_query = self.rewriter.rewrite(query)

        # --- Step 2: Mode Selection ---
        # Router uses the rewritten query (if available) for a more accurate decision
        if use_agent:
            selected_mode = self.agent.decide_mode(rewritten_query)
            agent_used = True
        else:
            selected_mode = mode   # use whatever the caller passed in
            agent_used = False

        # --- Step 3: Retrieval ---
        retrieved_docs = self.retrieval_service.search(
            query=rewritten_query ,
            mode=selected_mode,
            top_k=k,
        )

        # --- Step 4: Generation ---
        # Original query used here so the LLM answers what the user actually asked
        answer_text = self.generator_service.generate(
            query=query,
            documents=retrieved_docs,
        )

        return {
            "query": query,
            "rewritten_query": rewritten_query, # same as query if rewriter was off
            "mode": selected_mode,
            "agent_used": agent_used,
            "rewriter_used": rewriter_used,
            "retrieved_count": len(retrieved_docs),
            "answer": answer_text,
            "sources": retrieved_docs,
        }