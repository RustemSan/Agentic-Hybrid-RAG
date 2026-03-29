from fastapi import APIRouter, Query, HTTPException
from app.retrieval.search_client import SearchClient
from app.core.config import settings

router = APIRouter()
# Initializing the client. It will perform the health check we defined earlier.
search_service = SearchClient(host=settings.ELASTICSEARCH_HOST)

@router.get("/search")
async def search_questions(
    q: str = Query(..., min_length=3, description="Search query string"),
    limit: int = Query(5, ge=1, le=20)
):
    """
    Unified Search Endpoint for Flow A (BM25).
    Returns results strictly following the team's Retrieval Output Schema.
    """
    try:
        results = search_service.search(query=q, top_k=limit)
        return {
            "status": "success",
            "meta": {
                "query": q,
                "retrieved_count": len(results)
            },
            "data": results
        }
    except Exception as e:
        # If Elasticsearch is down or query is invalid
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")