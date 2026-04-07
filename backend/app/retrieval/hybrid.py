from backend.app.retrieval.search_client import SearchClient
from backend.app.retrieval.vector import VectorSearchClient


class HybridSearchClient:
    def __init__(
        self,
        bm25_client=None,
        vector_client=None,
        rrf_k: int = 60,
    ):
        self.bm25_client = bm25_client or SearchClient()
        self.vector_client = vector_client or VectorSearchClient()
        self.rrf_k = rrf_k

    def _doc_key(self, doc: dict):
        """
        Unique key for one retrieved document.
        In your dataset one QA pair is best identified by question_id.
        """
        return doc.get("question_id")

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    def search(self, query: str, top_k: int = 5):
        """
        Run BM25 and Vector retrieval, then merge results using RRF.
        Returns results in the same unified retrieval schema.
        """
        candidate_pool = max(top_k, 10)

        bm25_results = self.bm25_client.search(query, top_k=candidate_pool)
        vector_results = self.vector_client.search(query, top_k=candidate_pool)

        fused = {}

        # Add BM25 results
        for doc in bm25_results:
            key = self._doc_key(doc)
            if key is None:
                continue

            if key not in fused:
                fused[key] = {
                    "question_id": doc.get("question_id"),
                    "answer_id": doc.get("answer_id"),
                    "title": doc.get("title"),
                    "tags": doc.get("tags", []),
                    "question_text": doc.get("question_text"),
                    "answer_body": doc.get("answer_body"),
                    "combined_text": doc.get("combined_text"),
                    "retrieval_score": 0.0,
                    "retrieval_method": "hybrid",
                    "rank": None,
                    "bm25_score": None,
                    "vector_score": None,
                }

            fused[key]["retrieval_score"] += self._rrf_score(doc["rank"])
            fused[key]["bm25_score"] = doc.get("retrieval_score")

        # Add Vector results
        for doc in vector_results:
            key = self._doc_key(doc)
            if key is None:
                continue

            if key not in fused:
                fused[key] = {
                    "question_id": doc.get("question_id"),
                    "answer_id": doc.get("answer_id"),
                    "title": doc.get("title"),
                    "tags": doc.get("tags", []),
                    "question_text": doc.get("question_text"),
                    "answer_body": doc.get("answer_body"),
                    "combined_text": doc.get("combined_text"),
                    "retrieval_score": 0.0,
                    "retrieval_method": "hybrid",
                    "rank": None,
                    "bm25_score": None,
                    "vector_score": None,
                }

            fused[key]["retrieval_score"] += self._rrf_score(doc["rank"])
            fused[key]["vector_score"] = doc.get("retrieval_score")

        # Sort by fused RRF score descending
        ranked_results = sorted(
            fused.values(),
            key=lambda x: x["retrieval_score"],
            reverse=True,
        )

        # Reassign final rank
        for i, doc in enumerate(ranked_results, start=1):
            doc["rank"] = i

        return ranked_results[:top_k]