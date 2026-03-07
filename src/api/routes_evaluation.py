from fastapi import APIRouter

from .schemas import EvaluationRequest, EvaluationResponse
from ..evaluation.benchmark import run_benchmark

router = APIRouter()


@router.post("/", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):

    metrics = run_benchmark(
        dataset=request.dataset,
        top_k=request.top_k
    )

    return EvaluationResponse(
        recall_at_k=metrics["recall"],
        mrr=metrics["mrr"],
        avg_latency_ms=metrics["latency"]
    )