"""Health and system metadata route handlers."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from src.api.schemas import HealthResponse, MetadataResponse
from src.api.services import MarketPulseService

router = APIRouter(prefix="", tags=["System & Metadata"])


def get_service() -> MarketPulseService:
    """Dependency provider for MarketPulseService."""
    return MarketPulseService()


@router.get("/health", response_model=HealthResponse, summary="System Health Check")
def health_check() -> HealthResponse:
    """Returns the operational status, current version, and ISO timestamp."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/metadata", response_model=MetadataResponse, summary="Dataset & Hierarchy Metadata")
def get_system_metadata(service: MarketPulseService = Depends(get_service)) -> MetadataResponse:
    """Returns dataset dimensions, date span, available SKUs/stores, and the 12 M5 hierarchy levels."""
    return service.get_metadata()
