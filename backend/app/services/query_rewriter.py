class QueryRewriterAgent:
    def __init__(self, llm_client):
        if llm_client is None:
            raise ValueError("QueryRewriterAgent requires a valid llm_client, got None.")
        self.llm = llm_client

    def rewrite(self, query: str) -> str:
        prompt = f"""You are a search query optimizer for a technical Q&A retrieval system.

Rewrite the user query to improve document recall.
Rules:
- Expand abbreviations and informal phrasing
- Add relevant technical synonyms and related terms
- Keep it concise (max 15 words)
- Return ONLY the rewritten query, nothing else

Original query: {query}
Rewritten query:"""

        rewritten = self.llm.generate(prompt, max_tokens=40).strip()

        # fallback если что-то пошло не так
        if not rewritten or len(rewritten) < 3:
            return query

        return rewritten