from fastapi import APIRouter
from ..evaluation.benchmark import run_benchmark
from .schemas import EvaluationResponse, EvaluationRequest

router = APIRouter()

@router.post("/run", response_model=EvaluationResponse)
async def run_eval(request: EvaluationRequest):
    """
    Trigger a HotpotQA benchmark run.
    Note: this can take a minute as it pull datasets and generates embeddings.
    """
    results = await run_benchmark(sample_size=request.sample_size)
    
    return EvaluationResponse(
        recall_at_k=results["avg_recall_at_5"],
        mrr=results["mrr"],
        sample_size=results["sample_size"]
    )
