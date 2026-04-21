"""
vector.py — Semantic (dense vector) retrieval client using Qdrant.

Converts a text query into a dense embedding vector using SentenceTransformers,
then performs an approximate nearest-neighbor search in the Qdrant vector database
to find the most semantically similar documents.

Unlike BM25, vector search finds documents that are conceptually related
to the query even when they don't share exact keywords. For example:
  "how to iterate a collection" → matches documents about loops, generators, etc.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class VectorSearchClient:
    def __init__(
        self,
        host="localhost",
        port=6333,
        collection_name="stackoverflow_vector",
        model_name="BAAI/bge-small-en-v1.5",
    ):
        self.device = "cpu"
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        # SentenceTransformer is loaded once at init — not per query
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_query(self, query: str):
        # Convert a query string into a normalized dense embedding vector.
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def search(self, query: str, top_k: int = 5):
        # Embed the query and retrieve the top-k most similar documents from Qdrant.
        query_vector = self.embed_query(query)

        # with_payload=True ensures document fields are returned alongside the score
        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        formatted_results = []

        for i, hit in enumerate(response, start=1):
            payload = hit.payload or {}

            doc = {
                "question_id": payload.get("question_id"),
                "answer_id": payload.get("answer_id"),
                "title": payload.get("title"),
                "tags": payload.get("tags", []),
                "question_text": payload.get("question_text"),
                "answer_body": payload.get("answer_body"),
                "combined_text": payload.get("combined_text"),
                "retrieval_score": hit.score,   # cosine similarity (0.0–1.0)
                "retrieval_method": "vector",
                "rank": i,
            }
            formatted_results.append(doc)

        return formatted_results