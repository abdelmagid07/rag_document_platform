from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import time

from .schemas import QueryRequest, QueryResponse
from ..services.query_service import run_query, run_query_stream

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.time()
    answer, sources = await run_query(request.query, request.top_k)
    latency = (time.time() - start) * 1000
    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=latency
    )


@router.post("/stream")
async def query_stream(request: QueryRequest):
    """
    SSE streaming endpoint for RAG queries.
    """
    return StreamingResponse(
        run_query_stream(request.query, request.top_k),
        media_type="text/event-stream"
    )