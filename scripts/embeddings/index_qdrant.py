import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.core.config import settings

COLLECTION_NAME = "stackoverflow_vector"

VECTORS_PATH = Path("data/processed/embeddings/vectors.npy")
METADATA_PATH = Path("data/processed/embeddings/metadata.jsonl")

BATCH_SIZE = 500


def load_metadata(path: Path) -> list[dict]:
    metadata = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata

def make_point_id(idx: int, meta: dict) -> int:
    qid = meta.get("question_id", 0)
    aid = meta.get("answer_id", 0)
    return int(qid) * 10_000_000 + int(aid)

def main():
    collection_name = settings.QDRANT_COLLECTION
    qdrant_host = settings.QDRANT_HOST
    qdrant_port = settings.QDRANT_PORT

    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    print(f"Using collection: {collection_name}")

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
            f"Mismatch between vectors and metadata: "
            f"vectors={len(vectors)}, metadata={len(metadata)}"
        )

    vector_dim = int(vectors.shape[1])

    client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        timeout=300,
    )

    print("Recreating collection...")

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )

    print(f"Collection '{collection_name}' recreated.")

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
            id=make_point_id(idx, meta),
            vector=vector.tolist(),
            payload=payload,
        )

        points.append(point)

        if len(points) >= BATCH_SIZE:
            client.upsert(
                collection_name=collection_name,
                points=points,
            )
            total_uploaded += len(points)
            print(f"Uploaded {total_uploaded}/{len(metadata)} vectors...", flush=True)
            points = []

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        total_uploaded += len(points)

    print(f"Done. Uploaded {total_uploaded} vectors to '{collection_name}'.")

    collection_info = client.get_collection(collection_name=collection_name)
    print("Collection info:")
    print(collection_info)


if __name__ == "__main__":
    main()