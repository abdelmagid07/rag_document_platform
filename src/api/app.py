from fastapi import FastAPI

from .routes_documents import router as documents_router
from .routes_query import router as query_router
from .routes_evaluation import router as eval_router
from src.api.metrics import LatencyTracker

app = FastAPI(title="RAG Document Platform")
latency_tracker = LatencyTracker()

app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(eval_router, prefix="/evaluation", tags=["evaluation"])


@app.get("/status")
def status() -> dict:
    return {
        "service": "rag-document-platform",
        "status": "ok",
        "latency": latency_tracker.summary(),
    }


@app.get("/metrics")
def metrics() -> dict:
    return latency_tracker.summary()
