from fastapi import APIRouter, Query, HTTPException
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
from app.core.config import settings


router = APIRouter()
# Initializing the client. It will perform the health check we defined earlier.
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
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
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
        use_agent: bool = Query(default=False),
        use_rewriter: bool = Query(default=False),
):
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
                "rewritten_query": result["rewritten_query"],
                "mode": result["mode"],
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
        traceback.print_exc()
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