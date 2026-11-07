"""Hierarchical reconciliation route handlers."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import ReconcileRequest, ReconcileResponse, HierarchyLevelInfo
from src.api.services import MarketPulseService
from src.api.routes.health import get_service

router = APIRouter(prefix="/hierarchy", tags=["Hierarchical Reconciliation"])


@router.post("/reconcile", response_model=ReconcileResponse, summary="Execute Hierarchical Forecast Reconciliation")
def reconcile_forecasts(
    request: ReconcileRequest,
    service: MarketPulseService = Depends(get_service),
) -> ReconcileResponse:
    """Executes multi-level forecast reconciliation across all 12 hierarchy levels.

    Supported methods:
    - 'mint_shrink': Minimum Trace with Ledoit-Wolf shrinkage error covariance estimator.
    - 'bottom_up': Classic aggregation where higher levels are exactly S @ bottom_forecasts.
    - 'ols': Ordinary Least Squares reconciliation with diagonal identity covariance.
    """
    try:
        return service.reconcile_hierarchy(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reconciliation error: {str(exc)}") from exc


@router.get("/levels", response_model=List[HierarchyLevelInfo], summary="Get M5 Hierarchy Level Hierarchy")
def get_hierarchy_levels(
    service: MarketPulseService = Depends(get_service),
) -> List[HierarchyLevelInfo]:
    """Returns definitions and node counts for all 12 M5 hierarchical aggregation levels."""
    meta = service.get_metadata()
    return meta.hierarchy_levels
