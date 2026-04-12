# Agentic Hybrid RAG

## Overview

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) backend** combining:

* **BM25 retrieval** (Elasticsearch)
* **Vector retrieval** (Qdrant + embeddings)
* **Hybrid retrieval** (Reciprocal Rank Fusion)

The system retrieves relevant StackOverflow Q&A documents and exposes them via a unified FastAPI interface.

---

## Architecture

```
User Query
    ↓
FastAPI API
    ↓
RetrievalService
    ├── BM25 → Elasticsearch
    ├── Vector → Qdrant
    └── Hybrid → RRF(BM25 + Vector)
    ↓
Unified JSON Response
```

---

## Tech Stack

* **Backend:** FastAPI
* **BM25:** Elasticsearch
* **Vector DB:** Qdrant
* **Embeddings:** SentenceTransformers (`bge-small-en-v1.5`)
* **Dataset:** StackOverflow QA (Hugging Face: made by us from sof archive)
* **Language:** Python

---

## Setup

### 1. Start infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d elasticsearch qdrant
```

Check services:

```bash
curl http://127.0.0.1:9200
curl http://127.0.0.1:6333/collections
```

---

### 2. Run backend

```bash
cd backend
uvicorn app.main:app --reload
```

API docs:

```
http://127.0.0.1:8000/docs
```

---

## Data Indexing

### 1. BM25 (Elasticsearch)

```bash
PYTHONPATH=backend python scripts/ingest_data.py
```

Indexes ~10k documents into:

```
stackoverflow_bm25
```

---

### 2. Build embeddings (CPU)

```bash
PYTHONPATH=. python scripts/embeddings/build_embeddings.py
```

Outputs:

```
data/processed/embeddings/
  ├── vectors.npy
  ├── metadata.jsonl
  └── model_info.json
```

---

### 3. Index embeddings (Qdrant)

```bash
PYTHONPATH=backend python scripts/embeddings/index_qdrant.py
```

Creates collection:

```
stackoverflow_vector
```

---

## API Usage

### BM25

```bash
curl "http://127.0.0.1:8000/api/v1/search?q=python%20list&mode=bm25&limit=3"
```

### Vector

```bash
curl "http://127.0.0.1:8000/api/v1/search?q=python%20list&mode=vector&limit=3"
```

### Hybrid

```bash
curl "http://127.0.0.1:8000/api/v1/search?q=python%20list&mode=hybrid&limit=3"
```

---

## Output Format

All retrieval modes return the same schema:

```json
{
  "question_id": int,
  "answer_id": int,
  "title": string,
  "tags": [string],
  "question_text": string,
  "answer_body": string,
  "combined_text": string,
  "retrieval_score": float,
  "retrieval_method": "bm25 | vector | hybrid",
  "rank": int
}
```

---

## Hybrid Retrieval

Hybrid mode uses **Reciprocal Rank Fusion (RRF)**:

* Combines BM25 and vector rankings
* Improves robustness across query types
* Handles both keyword and semantic queries

---

## Important Notes

* Current setup uses **10k documents**
* Backend is currently run **locally (not fully dockerized)**

---

## Project Status

### Done

* BM25 indexing (Elasticsearch)
* Vector embeddings + Qdrant indexing
* Hybrid retrieval (RRF)
* Unified FastAPI API
* Working end-to-end retrieval pipeline

### Limitations

* No generation (LLM) layer yet
* No evaluation metrics
* Limited dataset size (10k)
* Minimal frontend
* Improve Docker setup
---

## Example Query

```
Query: "python list"
```

Hybrid retrieval returns:

* semantic matches (vector)
* keyword matches (BM25)
* fused ranking (hybrid)