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
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_query(self, query: str):
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def search(self, query: str, top_k: int = 5):
        query_vector = self.embed_query(query)

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
                "retrieval_score": hit.score,
                "retrieval_method": "vector",
                "rank": i,
            }
            formatted_results.append(doc)

        return formatted_results