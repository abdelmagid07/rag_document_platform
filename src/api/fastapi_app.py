from fastapi import FastAPI

from ..api.metrics import LatencyTracker

app = FastAPI(title="RAG Document Platform")
latency_tracker = LatencyTracker()


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
