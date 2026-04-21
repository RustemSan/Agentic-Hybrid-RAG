# Agentic Hybrid RAG

## Overview

This project implements an **Agentic Hybrid Retrieval-Augmented Generation (RAG) system** that combines:

* **BM25 retrieval** (Elasticsearch) — keyword-based search
* **Vector retrieval** (Qdrant + embeddings) — semantic search
* **Hybrid retrieval** (Reciprocal Rank Fusion) — best of both worlds
* **LLM generation** (OpenAI) — grounded answer generation over retrieved documents
* **Agent Router** — automatically selects the best retrieval mode per query
* **Query Rewriter** — expands and improves queries before retrieval to maximize recall

The system retrieves relevant StackOverflow Q&A documents, optionally rewrites the query and routes it through an intelligent agent, and generates a grounded technical answer via a FastAPI backend and Streamlit frontend.

---

## Architecture

```
User Query
    ↓
Streamlit Frontend
    ↓
FastAPI Backend 
    ↓
[Optional] Query Rewriter Agent  →  expands query for better recall
    ↓
[Optional] Router Agent          →  decides: bm25 / vector / hybrid
    ↓
RetrievalService
    ├── BM25   → Elasticsearch
    ├── Vector → Qdrant
    └── Hybrid → RRF(BM25 + Vector)
    ↓
GeneratorService (OpenAI LLM)
    ↓
Grounded Answer + Supporting Sources
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| BM25 | Elasticsearch |
| Vector DB | Qdrant |
| Embeddings | SentenceTransformers (`bge-small-en-v1.5`) |
| Generation | OpenAI API |
| Agents | Custom LLM-based routing + rewriting |
| Dataset | StackOverflow QA (custom, ~10k docs) |
| Language | Python 3.11+ |

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── router.py              # FastAPI endpoints
│   │   ├── core/
│   │   │   └── config.py              # Settings (env vars)
│   │   ├── retrieval/
│   │   │   ├── search_client.py       # BM25 / Elasticsearch client
│   │   │   ├── vector.py              # Vector / Qdrant client
│   │   │   └── hybrid.py              # Hybrid RRF client
│   │   └── services/
│   │       ├── retrieval_service.py   # Unified retrieval interface
│   │       ├── generator_service.py   # LLM answer generation
│   │       ├── rag_service.py         # Orchestrates retrieval + generation
│   │       ├── llm_client.py          # Thin OpenAI wrapper for agents
│   │       ├── agent_router.py        # Router agent (mode selection)
│   │       └── query_rewriter.py      # Rewriter agent (query expansion)
│   └── main.py
├── frontend/
│   └── app.py                         # Streamlit UI
├── scripts/
│   ├── ingest_data.py                 # Index data into Elasticsearch
│   └── embeddings/
│       ├── build_embeddings.py        # Build vector embeddings
│       └── index_qdrant.py            # Index embeddings into Qdrant
├── docker/
│   └── docker-compose.yml
└── data/
    └── processed/
        └── embeddings/
            ├── vectors.npy
            ├── metadata.jsonl
            └── model_info.json
```

---

## Setup

### 1. Environment variables

Create a `.env` file in `backend/`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

ELASTICSEARCH_HOST=http://127.0.0.1:9200
INDEX_NAME=stackoverflow_bm25

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=stackoverflow_vector

EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5

DEFAULT_RETRIEVAL_MODE=hybrid
DEFAULT_TOP_K=5
GENERATION_TOP_K=4
MAX_TOP_K=10
GENERATION_MAX_OUTPUT_TOKENS=1024
GENERATION_TEMPERATURE=0.2
```

---

### 2. Start infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d elasticsearch qdrant
```

Verify services are running:

```bash
curl http://127.0.0.1:9200          # Elasticsearch
curl http://127.0.0.1:6333/collections  # Qdrant
```

---

### 3. Index data

```bash
# BM25 index (Elasticsearch)
PYTHONPATH=backend python scripts/ingest_data.py

# Build vector embeddings (CPU, takes a few minutes)
PYTHONPATH=. python scripts/embeddings/build_embeddings.py

# Index embeddings into Qdrant
PYTHONPATH=backend python scripts/embeddings/index_qdrant.py
```

---

### 4. Run backend

```bash
cd backend
uvicorn app.main:app --reload
```

API docs available at: `http://127.0.0.1:8000/docs`

---

### 5. Run frontend

```bash
cd frontend
streamlit run app.py
```

Frontend available at: `http://127.0.0.1:8501`

---

## API Endpoints

### `GET /api/v1/search` — Document retrieval only

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Search query |
| `mode` | string | `hybrid` | `bm25`, `vector`, or `hybrid` |
| `limit` | int | `5` | Number of documents to return (1–100) |

```bash
curl "http://127.0.0.1:8000/api/v1/search?q=python+list&mode=hybrid&limit=3"
```

---

### `GET /api/v1/answer` — Retrieval + LLM generation

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | User question |
| `mode` | string | `hybrid` | Retrieval mode (ignored if `use_agent=true`) |
| `limit` | int | `4` | Number of documents to pass to LLM |
| `use_agent` | bool | `false` | Let the Router Agent choose retrieval mode |
| `use_rewriter` | bool | `false` | Rewrite query before retrieval |

```bash
# Manual mode
curl "http://127.0.0.1:8000/api/v1/answer?q=how+to+use+generators+in+python&mode=hybrid&limit=4"

# With both agents enabled
curl "http://127.0.0.1:8000/api/v1/answer?q=how+list+works+python&use_agent=true&use_rewriter=true"
```

---

## Agents

### Router Agent (`agent_router.py`)

Decides which retrieval mode to use based on query type:

| Query type | Selected mode |
|---|---|
| Factual, definition-based | `bm25` |
| Conceptual, semantic | `vector` |
| Mixed or unclear | `hybrid` |

Enable via: `use_agent=true`

---

### Query Rewriter Agent (`query_rewriter.py`)

Expands informal or short queries into richer technical queries before retrieval.

```
"how list works python"
    → "python list operations methods mutability indexing"
```

Enable via: `use_rewriter=true`

Both agents can be used together — the rewriter runs first, then the router decides mode based on the rewritten query.

---

## Response Format

### `/api/v1/answer` response

```json
{
  "status": "success",
  "meta": {
    "query": "how list works python",
    "rewritten_query": "python list operations methods mutability indexing",
    "mode": "vector",
    "agent_used": true,
    "rewriter_used": true,
    "retrieved_count": 4,
    "limit": 4
  },
  "data": {
    "answer": "A Python list is a mutable, ordered collection...",
    "sources": [ ... ]
  }
}
```

### Document schema (retrieval results)

```json
{
  "question_id": 1234,
  "answer_id": 5678,
  "title": "How do Python lists work?",
  "tags": ["python", "list"],
  "question_text": "...",
  "answer_body": "...",
  "combined_text": "...",
  "retrieval_score": 0.0312,
  "retrieval_method": "hybrid",
  "rank": 1
}
```

---

## Project Status

### Done

- BM25 indexing (Elasticsearch)
- Vector embeddings + Qdrant indexing
- Hybrid retrieval (Reciprocal Rank Fusion)
- Unified FastAPI retrieval API
- LLM generation layer (OpenAI)
- Router Agent (automatic mode selection)
- Query Rewriter Agent (query expansion)
- Streamlit frontend with both agents toggleable

### Limitations / TODO

- No evaluation metrics (recall@k, MRR, faithfulness)
- Limited dataset size (~10k docs)
- Backend not fully dockerized (runs locally)
- No streaming generation
- No conversation history / multi-turn support