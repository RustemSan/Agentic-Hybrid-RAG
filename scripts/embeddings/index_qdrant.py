import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION_NAME = "stackoverflow_vector"

VECTORS_PATH = Path("data/processed/embeddings/vectors.npy")
METADATA_PATH = Path("data/processed/embeddings/metadata.jsonl")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

BATCH_SIZE = 500


def load_metadata(path: Path) -> list[dict]:
    metadata = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def main():
    if not VECTORS_PATH.exists():
        raise FileNotFoundError(f"Vectors file not found: {VECTORS_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    print("Loading vectors...")
    vectors = np.load(VECTORS_PATH)
    print(f"Vectors loaded. Shape: {vectors.shape}")

    print("Loading metadata...")
    metadata = load_metadata(METADATA_PATH)
    print(f"Metadata loaded. Count: {len(metadata)}")

    if len(vectors) != len(metadata):
        raise ValueError(
            f"Mismatch: vectors={len(vectors)} metadata={len(metadata)}"
        )

    vector_dim = int(vectors.shape[1])

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)

    print("Recreating collection...")

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )

    print(f"Collection '{COLLECTION_NAME}' recreated.")

    points = []
    total_uploaded = 0

    for idx, (vector, meta) in enumerate(zip(vectors, metadata)):
        payload = {
            "question_id": meta.get("question_id"),
            "answer_id": meta.get("answer_id"),
            "title": meta.get("title"),
            "tags": meta.get("tags", []),
            "question_text": meta.get("question_text"),
            "answer_body": meta.get("answer_body"),
            "combined_text": meta.get("combined_text"),
        }

        point = PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload=payload,
        )

        points.append(point)

        if len(points) >= BATCH_SIZE:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            total_uploaded += len(points)
            if total_uploaded % 10000 == 0:
                print(f"Uploaded {total_uploaded} vectors...", flush=True)
            points = []

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        total_uploaded += len(points)

    print(f"Done. Uploaded {total_uploaded} vectors to '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()