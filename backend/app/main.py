from fastapi import FastAPI
from app.api.router import router as search_router

app = FastAPI(
    title="Agentic-Hybrid-RAG Backend",
    description="API for StackOverflow search and AI Agent interaction",
    version="0.1.0"
)

# Registering the search routes under /api/v1 prefix
app.include_router(search_router, prefix="/api/v1", tags=["Retrieval"])

@app.get("/health")
def health_check():
    """Simple health check for Docker/Monitoring"""
    return {"status": "healthy", "service": "backend"}

if __name__ == "__main__":
    import uvicorn
    # Allows running the server directly: python app/main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)