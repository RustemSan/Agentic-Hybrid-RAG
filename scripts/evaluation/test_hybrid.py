from backend.app.retrieval.hybrid import HybridSearchClient


def main():
    client = HybridSearchClient()

    queries = [
        "convert decimal to double c#",
        "calculate age from datetime c#",
        "ie7 percentage width div problem",
        "python import module error",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}\n")

        results = client.search(query, top_k=5)

        for r in results:
            print(f"Rank: {r['rank']}")
            print(f"Hybrid score: {r['retrieval_score']:.6f}")
            print(f"BM25 score: {r['bm25_score']}")
            print(f"Vector score: {r['vector_score']}")
            print(f"Title: {r['title']}")
            print(f"Tags: {r['tags']}")
            print("-" * 80)


if __name__ == "__main__":
    main()