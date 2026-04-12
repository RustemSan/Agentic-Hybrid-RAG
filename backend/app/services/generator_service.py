from __future__ import annotations

from typing import Any, Dict, List

import requests

from app.core.config import settings


class GeneratorService:
    def __init__(self) -> None:
        self.api_url = settings.LLM_API_URL
        self.model_name = settings.LLM_MODEL_NAME
        self.api_key = settings.LLM_API_KEY
        self.max_tokens = settings.GENERATION_MAX_TOKENS
        self.temperature = settings.GENERATION_TEMPERATURE
        self.timeout = settings.REQUEST_TIMEOUT_SEC

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

    def build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a retrieval-grounded technical assistant. "
            "Answer only using the provided context. "
            "If the context is insufficient, say that the available documents are insufficient. "
            "Be precise and concise. "
            "When possible, synthesize the answer rather than copying text."
        )

        user_prompt = (
            f"User question:\n{query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Write a helpful technical answer grounded in the retrieved documents."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate(self, query: str, documents: List[Dict[str, Any]]) -> str:
        context = self.build_context(documents)
        messages = self.build_messages(query=query, context=context)

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()