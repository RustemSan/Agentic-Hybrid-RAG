"""
router.py — FastAPI endpoints for the RAG system.

Exposes three groups of routes:
  - /search         — raw document retrieval (no generation)
  - /answer         — full RAG pipeline: retrieval + LLM answer generation
  - /search/modes   — metadata about supported retrieval modes
  - /search/health  — lightweight health check for retrieval backends
"""

from fastapi import APIRouter, Query, HTTPException
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
from app.core.config import settings


router = APIRouter()

# Both services are initialized once at module load time (not per-request)
# RetrievalService connects to Elasticsearch and Qdrant
# RAGService wraps retrieval + generation + agent
retrieval_service = RetrievalService()
rag_service = RAGService(retrieval_service=retrieval_service)

@router.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: str = Query(
        default=settings.DEFAULT_RETRIEVAL_MODE,
        description="Retrieval mode: bm25, vector, hybrid"
    ),
    limit: int = Query(
        default=settings.DEFAULT_TOP_K,
        ge=1,
        le=settings.MAX_TOP_K,
        description="Number of results to return"
    ),
):
    """
    Retrieve documents without generating an answer.

    Useful for inspecting what the retrieval pipeline returns
    before committing to full generation.

    Returns a list of ranked documents with scores and metadata.
    """
    try:
        results = retrieval_service.search(
            query=q,
            mode=mode,
            top_k=limit,
        )

        return {
            "status": "success",
            "meta": {
                "query": q,
                "mode": mode,
                "retrieved_count": len(results),
                "limit": limit,
            },
            "data": results,
        }

    except ValueError as e:
        # Triggered by invalid mode, empty query, or bad top_k value
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        # Triggered when Elasticsearch or Qdrant call fails
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(e)}"
        )


@router.get("/answer")
def answer(
    q: str = Query(..., min_length=1, description="User question"),
    mode: str = Query(
        default=settings.DEFAULT_RETRIEVAL_MODE,
        description="Retrieval mode: bm25, vector, hybrid"
    ),
    limit: int = Query(
        default=settings.GENERATION_TOP_K,
        ge=1,
        le=settings.MAX_TOP_K,
        description="Number of retrieved documents to use for generation"
    ),
        use_agent: bool = Query(
            default=False,
            description="If true, Router Agent selects the retrieval mode automatically",
        ),
        use_rewriter: bool = Query(
            default=False,
            description="If true, Query Rewriter expands the query before retrieval",
        ),

):
    """
        Full RAG pipeline: retrieve documents and generate a grounded answer.

        Pipeline order when both agents are enabled:
          1. Query Rewriter expands the query for better recall
          2. Router Agent selects the best retrieval mode
          3. Retrieval runs against the selected backend
          4. LLM generates an answer grounded in the retrieved documents

        If use_agent=False, the `mode` parameter is used directly.
        If use_rewriter=False, the original query is passed to retrieval as-is.

        Returns the generated answer, agent metadata, and the source documents used.
        """
    try:
        result = rag_service.answer(
            query=q,
            mode=mode,
            top_k=limit,
            use_agent=use_agent,
            use_rewriter=use_rewriter,
        )

        return {
            "status": "success",
            "meta": {
                "query": result["query"],
                "rewritten_query": result["rewritten_query"], # equals query if rewriter was off
                "mode": result["mode"],  # actual mode used (may differ if agent was on)
                "agent_used": result["agent_used"],
                "rewriter_used": result["rewriter_used"],
                "retrieved_count": result["retrieved_count"],
                "limit": limit,
            },
            "data": {
                "answer": result["answer"],
                "sources": result["sources"],
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full stack trace to server console for debugging
        raise HTTPException(status_code=500, detail=f"Unexpected generation error: {str(e)}")




@router.get("/search/modes")
def get_search_modes():
    return {
        "status": "success",
        "data": {
            "supported_modes": retrieval_service.get_supported_modes(),
            "default_mode": settings.DEFAULT_RETRIEVAL_MODE,
            "default_top_k": settings.DEFAULT_TOP_K,
            "max_top_k": settings.MAX_TOP_K,
        },
    }


@router.get("/search/health")
def search_health():
    try:
        return {
            "status": "success",
            "data": retrieval_service.healthcheck(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval service healthcheck failed: {str(e)}"
        )