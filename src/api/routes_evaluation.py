from fastapi import APIRouter
from ..evaluation.benchmark import run_benchmark
from .schemas import EvaluationResponse, EvaluationRequest

router = APIRouter()

@router.post("/run", response_model=EvaluationResponse)
async def run_eval(request: EvaluationRequest):
    """
    Trigger a HotpotQA benchmark run.
    """
    try:
        results = await run_benchmark(sample_size=request.sample_size)
        return EvaluationResponse(
            recall_at_k=results["avg_recall_at_5"],
            mrr=results["mrr"],
            sample_size=results["sample_size"]
        )
    except Exception as e:
        from ..services.logger import logger
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        # Re-raise to let FastAPI handle it but we've logged it
        raise
