"""Production drift monitoring and retraining governance route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Query
from src.api.schemas import DriftAnalysisResponse, RetrainingDecisionResponse
from src.api.services import MarketPulseService
from src.api.routes.health import get_service

router = APIRouter(prefix="/monitoring", tags=["Production Monitoring & Governance"])


@router.get("/drift", response_model=DriftAnalysisResponse, summary="Inspect Covariate & Target Distribution Drift")
def get_drift_analysis(
    window: str = Query("Window_2", description="Monitoring window ID ('Window_1' or 'Window_2')"),
    service: MarketPulseService = Depends(get_service),
) -> DriftAnalysisResponse:
    """Returns Population Stability Index (PSI), two-sample Kolmogorov-Smirnov statistics,

    Earth Mover's Distance (Wasserstein), and demand regime shift metrics.
    """
    try:
        return service.get_drift_analysis(window_id=window)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf)) from fnf
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift analysis error: {str(exc)}") from exc


@router.get(
    "/retraining-decision",
    response_model=RetrainingDecisionResponse,
    summary="Evaluate Retraining Justification Matrix",
)
def get_retraining_decision(
    window: str = Query("window_2", description="Evaluation window ID ('window_1' or 'window_2')"),
    service: MarketPulseService = Depends(get_service),
) -> RetrainingDecisionResponse:
    """Returns objective 4-tier model retraining decision (NO_ACTION, WARNING,

    RETRAIN_RECOMMENDED, CRITICAL_FALLBACK) with statistical justifications.
    """
    try:
        return service.get_retraining_decision(window_id=window)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf)) from fnf
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining evaluation error: {str(exc)}") from exc
