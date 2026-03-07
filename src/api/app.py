from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()  
from .routes_documents import router as documents_router
from .routes_query import router as query_router
from .metrics import LatencyTracker

app = FastAPI(
    title="RAG Document Platform",
)

latency_tracker = LatencyTracker()

app.include_router(documents_router, prefix="/documents", tags=["Documents"])
app.include_router(query_router, prefix="/query", tags=["Query"])


@app.get("/", tags=["Health"])
def root():
    return {"message": "RAG Document Platform API", "docs": "/docs"}


@app.get("/status", tags=["Health"])
def status() -> dict:
    return {
        "service": "rag-document-platform",
        "status": "ok",
        "latency": latency_tracker.summary(),
    }


@app.get("/metrics", tags=["Health"])
def metrics() -> dict:
    return latency_tracker.summary()
