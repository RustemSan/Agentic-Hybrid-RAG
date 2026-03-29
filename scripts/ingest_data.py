import sys
import os
from datasets import load_dataset
from tqdm import tqdm
import time
from app.core.config import settings
from elasticsearch import exceptions
# Adding the 'backend' folder to the path so we can import our SearchClient
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from app.retrieval.search_client import SearchClient


def wait_for_elastic(client, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if client.es.ping():
                print("ElasticSearch is up and running!")
                return True
        except exceptions.ConnectionError:
            pass
        print("Waiting for ElasticSearch to wake up...")
        time.sleep(5)
    raise Exception("ElasticSearch is not responding after timeout")


def main():

    print(f"Connecting to ElasticSearch at: {settings.ELASTICSEARCH_HOST}")

    client = SearchClient(host=settings.ELASTICSEARCH_HOST)
    wait_for_elastic(client)

    # 1. Initialize the index (deletes and recreates if needed is handled in client or manually)
    print("Initializing ElasticSearch Index")
    client.create_index()

    print("Streaming dataset from Hugging Face...")
    dataset = load_dataset("krylodar/StackOverFlowQA", split="train", streaming=True)

    batch = []
    batch_size = 200
    limit = 10000

    print(f"Starting ingestion (limit {limit} docs)...")
    for i, row in tqdm(enumerate(dataset), total=limit):
        if i >= limit:
            break

        if i == 0:
            print(f"DEBUG: Available keys in row: {row.keys()}")
        # Mapping dataset fields to your team's Unified Schema
        doc = {
            "question_id": row.get("qid") or row.get("question_id"),
            "answer_id": row.get("aid") or row.get("answer_id"),
            "title": row.get("title"),
            "tags": row.get("tags", []),
            "question_text": row.get("question_body"),
            "answer_body": row.get("answer_body"),
            "combined_text": row.get("combined_text")
        }

        batch.append(doc)

        # Bulk upload when batch is full
        if len(batch) >= batch_size:
            client.bulk_index(batch)
            batch = []
            time.sleep(0.5)

    if batch:
        client.bulk_index(batch)

    print("\nIngestion complete!")



if __name__ == "__main__":
    main()


