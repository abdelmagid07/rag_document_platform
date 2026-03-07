from fastapi import APIRouter
import time

from .schemas import QueryRequest, QueryResponse
from ..services.query_service import run_query

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