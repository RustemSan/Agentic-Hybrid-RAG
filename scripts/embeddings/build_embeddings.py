import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


OUTPUT_DIR = Path("data/processed/embeddings")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64
LIMIT = 10000   # ДОЛЖЕН совпадать с BM25 ingestion


def stream_documents(limit: int):
    """
    Stream documents from Hugging Face dataset
    using the same unified schema as Elasticsearch ingestion.
    """
    dataset = load_dataset("krylodar/StackOverFlowQA", split="train", streaming=True)

    docs = []
    for i, row in tqdm(enumerate(dataset), total=limit, desc="Streaming dataset"):
        if i >= limit:
            break

        doc = {
            "question_id": row.get("qid") or row.get("question_id"),
            "answer_id": row.get("aid") or row.get("answer_id"),
            "title": row.get("title"),
            "tags": row.get("tags", []),
            "question_text": row.get("question_body") or row.get("question_text"),
            "answer_body": row.get("answer_body"),
            "combined_text": row.get("combined_text"),
        }

        docs.append(doc)

    return docs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading documents from Hugging Face...")
    docs = stream_documents(LIMIT)
    print(f"Loaded {len(docs)} documents")

    texts = [doc["combined_text"] for doc in docs if doc.get("combined_text")]

    if len(texts) != len(docs):
        raise ValueError(
            f"Mismatch: docs={len(docs)}, texts_with_combined_text={len(texts)}. "
            "Some documents are missing combined_text."
        )

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding texts...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Vectors shape: {vectors.shape}")

    np.save(OUTPUT_DIR / "vectors.npy", vectors)

    with (OUTPUT_DIR / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for doc in docs:
            meta = {
                "question_id": doc["question_id"],
                "answer_id": doc["answer_id"],
                "title": doc["title"],
                "tags": doc["tags"],
                "question_text": doc["question_text"],
                "answer_body": doc["answer_body"],
                "combined_text": doc["combined_text"],
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    model_info = {
        "model_name": MODEL_NAME,
        "dataset_name": "krylodar/StackOverFlowQA",
        "num_documents": len(docs),
        "embedding_dim": int(vectors.shape[1]),
        "normalized": True,
        "text_field": "combined_text",
        "limit": LIMIT,
    }

    with (OUTPUT_DIR / "model_info.json").open("w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings for {len(docs)} documents.")
    print(f"Files saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()