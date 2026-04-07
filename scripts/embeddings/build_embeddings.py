import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/docs100k.jsonl")
OUTPUT_DIR = Path("data/processed/embeddings")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def load_documents(path: Path):
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading documents...")

    docs = load_documents(INPUT_PATH)
    print("Docs are loaded")
    texts = [doc["combined_text"] for doc in docs]

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print("model is connected")

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

    print("Docs are saved")

    model_info = {
        "model_name": MODEL_NAME,
        "input_file": str(INPUT_PATH),
        "num_documents": len(docs),
        "embedding_dim": int(vectors.shape[1]),
        "normalized": True,
        "text_field": "combined_text",
    }

    with (OUTPUT_DIR / "model_info.json").open("w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings for {len(docs)} documents.")
    print(f"Vector shape: {vectors.shape}")


if __name__ == "__main__":
    main()