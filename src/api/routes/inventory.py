"""Inventory optimization and Monte Carlo simulation route handlers."""

from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import (
    InventoryPolicyRequest,
    InventoryPolicyResponse,
    InventorySimulationRequest,
    InventorySimulationResponse,
)
from src.api.services import MarketPulseService
from src.api.routes.health import get_service

router = APIRouter(prefix="/inventory", tags=["Inventory Decision Engine"])


@router.post("/policy", response_model=InventoryPolicyResponse, summary="Optimize Inventory Replenishment Policy")
def optimize_inventory_policy(
    request: InventoryPolicyRequest,
    service: MarketPulseService = Depends(get_service),
) -> InventoryPolicyResponse:
    """Calculates mathematically optimal Safety Stock (SS), Reorder Point (ROP),

    Economic Order Quantity (EOQ), and Target Stock Level (S) for continuous (s, S) or (R, s, S) review.
    """
    try:
        return service.compute_inventory_policy(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Policy calculation error: {str(exc)}") from exc


@router.post("/simulate", response_model=InventorySimulationResponse, summary="Discrete-Event Monte Carlo Simulation")
def simulate_inventory(
    request: InventorySimulationRequest,
    service: MarketPulseService = Depends(get_service),
) -> InventorySimulationResponse:
    """Simulates daily inventory transitions (On-Hand, On-Order, Stockouts, Replenishments)

    under stochastic customer demand and lead-time variability.
    Returns supply chain KPIs: Fill Rate, Cycle Service Level (CSL), Inventory Turns, DIO, and Total Cost.
    """
    try:
        return service.simulate_inventory(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(exc)}") from exc
