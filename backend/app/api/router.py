from fastapi import APIRouter, Query, HTTPException
from app.retrieval_service import RetrievalService
from app.core.config import settings

router = APIRouter()
# Initializing the client. It will perform the health check we defined earlier.
retrieval_service = RetrievalService()

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