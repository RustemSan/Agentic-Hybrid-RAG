"""
agent_router.py — LLM-based retrieval mode selector.

This agent decides which retrieval backend to use for a given query
by asking the LLM to classify the query type. This removes the need
for the user to manually pick a mode.

Possible outputs: "bm25", "vector", "hybrid"
Falls back to "hybrid" if the LLM returns anything unexpected.
"""

class RetrievalRouterAgent:
    VALID_MODES = {"BM25", "VECTOR", "HYBRID"}

    def __init__(self, llm_client):
        if llm_client is None:
            raise ValueError("RetrievalRouterAgent requires a valid llm_client, got None.")
        self.llm = llm_client

    def decide_mode(self, query: str) -> str:
        prompt = f"""You are a routing system for a RAG pipeline.

Choose ONE retrieval mode:
- bm25: keyword-based, factual, definition questions
- vector: semantic, conceptual questions  
- hybrid: mixed or unclear

Return only one word: bm25, vector, or hybrid.

Query: {query}"""

        response = self.llm.generate(prompt).strip().upper()

        if response not in self.VALID_MODES:
            return "hybrid"  # fallback

        return response