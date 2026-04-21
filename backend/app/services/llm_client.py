"""
llm_client.py — Thin wrapper around the OpenAI client for agent use.

GeneratorService makes its own OpenAI calls directly (full RAG generation).
This client exists separately for the agents (Router, Rewriter) which need
short, low-latency completions with different parameters (low max_tokens,
temperature=0 for deterministic routing/rewriting decisions).

Keeping this separate means agents don't need to know about OpenAI internals —
they just call llm_client.generate(prompt) and get a string back.
"""


class LLMClient:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def generate(self, prompt: str, max_tokens=20, temp=0.0):
        return self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temp,
            max_output_tokens=max_tokens,
        ).output_text.strip()