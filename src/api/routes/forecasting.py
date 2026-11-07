"""Demand forecasting route handlers."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import ForecastRequest, ForecastResponse
from src.api.services import MarketPulseService
from src.api.routes.health import get_service

router = APIRouter(prefix="", tags=["Demand Forecasting"])


@router.post("/forecast", response_model=ForecastResponse, summary="Generate Demand Forecast")
def generate_forecast(
    request: ForecastRequest,
    service: MarketPulseService = Depends(get_service),
) -> ForecastResponse:
    """Generates point forecasts and probabilistic prediction intervals ($q \\in [0.05, 0.95]$)

    Supports:
    - Recursive LightGBM Tabular Forecaster
    - AutoARIMA with seasonal periodicity (sp=7)
    - Croston method for intermittent/lumpy series
    - Seasonal Naive baseline
    """
    try:
        return service.generate_forecast(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecasting engine error: {str(exc)}") from exc


@router.get("/models", response_model=List[str], summary="List Supported Model Families")
def list_available_models() -> List[str]:
    """Returns candidate forecasting algorithms supported by the serving engine."""
    return ["lightgbm", "auto_arima", "croston", "seasonal_naive"]
