"""
query_rewriter.py — LLM-based query expansion agent.

Short or informal user queries often miss relevant documents because the exact
keywords don't match. This agent rewrites the query before retrieval to:
  - expand abbreviations ("py" → "python")
  - add technical synonyms ("how list works" → "python list operations mutability")
  - normalize informal phrasing into precise technical terms

The rewriter runs before both the Router Agent and the retrieval step,
so the improved query benefits both mode selection and document matching.
"""

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

        # Sanity check: if LLM returns empty string or garbage, use original query
        if not rewritten or len(rewritten) < 3:
            return query

        return rewritten