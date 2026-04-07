from backend.app.retrieval.vector import VectorSearchClient
from backend.app.retrieval.search_client import SearchClient


def print_results(title, results):
    print(f"\n--- {title} ---")
    for r in results:
        print(f"{r['rank']}. {r['title']} (score={r['retrieval_score']:.4f})")


def main():
    bm25 = SearchClient()
    vector = VectorSearchClient()

    queries = [
        "convert decimal to double c#",
        "calculate age from datetime c#",
        "ie7 div width problem",
        "python import module error",
        "how to center div css",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")

        bm25_results = bm25.search(query, top_k=5)
        vector_results = vector.search(query, top_k=5)

        print_results("BM25", bm25_results)
        print_results("VECTOR", vector_results)


if __name__ == "__main__":
    main()