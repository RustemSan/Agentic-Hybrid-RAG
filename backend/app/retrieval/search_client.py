from elasticsearch import Elasticsearch, helpers

class SearchClient:
    def __init__(self, host="http://localhost:9200"):
        "ElasticSearch connection to Docker"
        self.es = Elasticsearch(
            host,
            request_timeout=60,
            retry_on_timeout=True,
            max_retries=3
        )
        self.index_name = "stackoverflow_bm25"

    def create_index(self):
        "Creating index with mapping for Comparison Format"
        settings = {
            "mappings": {
                "properties": {
                    "question_id": {"type": "long"},
                    "answer_id": {"type": "long"},
                    "title": {"type": "text", "analyzer": "english"},
                    "tags": {"type": "keyword"},
                    "question_text": {"type": "text", "analyzer": "english"},
                    "answer_body": {"type": "text", "analyzer": "english"},
                    "combined_text": {"type": "text", "analyzer": "english"}
                }
            }
        }

        # If index already created its better to create again because of mapping
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=settings)
            print(f"Index {self.index_name} was succesfully created")

    def bulk_index(self, docs: list):
        """
        Index multiple documents efficiently using the Bulk API.
        :param docs: List of dictionaries containing StackOverflow data.
        """
        actions = [
            {
                "_index": self.index_name,
                "_source": doc
            }
            for doc in docs
        ]
        helpers.bulk(self.es, actions)


    def search(self, query: str, top_k: int = 5):
        """
        Execute a BM25 search and return results formatted according to the retrieval output schema.
        :param query: The search string.
        :param top_k: Number of results to return.
        """
        search_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "combined_text"] # Title has 3x higher priority
                }
            }
        }

        response = self.es.search(index=self.index_name, body=search_query, size=top_k)

        formatted_results = []
        # Enumerate to generate the 'rank' field starting from 1
        for i, hit in enumerate(response['hits']['hits'], start=1):
            source = hit['_source']

            # Map Elasticsearch hit to the unified retrieval format
            doc = {
                "question_id": source.get("question_id"),
                "answer_id": source.get("answer_id"),
                "title": source.get("title"),
                "tags": source.get("tags", []),
                "question_text": source.get("question_text"),
                "answer_body": source.get("answer_body"),
                "combined_text": source.get("combined_text"),
                "retrieval_score": hit["_score"],
                "retrieval_method": "bm25",
                "rank": i
            }
            formatted_results.append(doc)

        return formatted_results

