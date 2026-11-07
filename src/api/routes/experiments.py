"""Research experiments benchmark results route handler."""

from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import BenchmarkResultsResponse
from src.api.services import MarketPulseService
from src.api.routes.health import get_service

router = APIRouter(prefix="/experiments", tags=["Research Benchmarks"])


@router.get("/benchmarks", response_model=BenchmarkResultsResponse, summary="Explore Research Experiment Benchmarks")
def get_benchmark_results(
    service: MarketPulseService = Depends(get_service),
) -> BenchmarkResultsResponse:
    """Returns summarized results for all 6 core research experiments:

    1. Global LightGBM vs. Local ARIMA
    2. Value of Hierarchical Reconciliation
    3. Exogenous Feature Ablation Study
    4. Cold-Start SKU Transfer Learning
    5. Demand Regime Dynamic Model Routing
    6. Multi-Horizon Error Deterioration
    """
    try:
        return service.get_benchmark_results()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark retrieval error: {str(exc)}") from exc
