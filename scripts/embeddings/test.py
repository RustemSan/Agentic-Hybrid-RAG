from backend.app.retrieval.vector import VectorSearchClient
from qdrant_client import QdrantClient

def main():
    cl = QdrantClient(host="localhost", port=6333)

    print(cl.get_collection("stackoverflow_vector"))
    print(cl.count("stackoverflow_vector", exact=True))


    client = VectorSearchClient()

    queries = [
        "convert decimal to double c#",
        "calculate age from datetime c#",
        "ie7 percentage width div problem",
        "python import error module not found",
    ]

    for query in queries:
        print("=" * 80)
        print(f"QUERY: {query}\n")

        results = client.search(query, top_k=5)

        for r in results:
            print(f"Rank: {r['rank']}")
            print(f"Score: {r['retrieval_score']:.4f}")
            print(f"Title: {r['title']}")
            print(f"Tags: {r['tags']}")
            print("-" * 80)


if __name__ == "__main__":
    main()