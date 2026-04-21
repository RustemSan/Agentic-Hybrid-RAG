from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from app.core.config import settings


class GeneratorService:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = settings.OPENAI_MODEL
        self.max_output_tokens = settings.GENERATION_MAX_OUTPUT_TOKENS
        self.temperature = settings.GENERATION_TEMPERATURE

    def build_context(self, documents: List[Dict[str, Any]]) -> str:
        context_blocks = []

        for i, doc in enumerate(documents, start=1):
            block = (
                f"[Document {i}]\n"
                f"Question ID: {doc.get('question_id')}\n"
                f"Answer ID: {doc.get('answer_id')}\n"
                f"Title: {doc.get('title')}\n"
                f"Tags: {', '.join(doc.get('tags', []))}\n"
                f"Question: {doc.get('question_text')}\n"
                f"Answer: {doc.get('answer_body')}\n"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def build_input(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a retrieval-grounded technical assistant. "
            "Answer only using the provided context. "
            "If the context is insufficient, clearly say that the available documents are insufficient. "
            "Be precise, concise, and technically correct. "
            "Do not invent facts outside the retrieved documents."
        )

        user_prompt = (
            f"User question:\n{query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Write a grounded technical answer based only on the retrieved context."
        )

        return f"{system_prompt}\n\n{user_prompt}"

    def generate(self, query: str, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "The available documents are insufficient to answer the question."

        context = self.build_context(documents)
        full_input = self.build_input(query=query, context=context)

        response = self.client.responses.create(
            model=self.model_name,
            input=full_input,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        return response.output_text.strip()

