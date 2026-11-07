"""API route module exports."""

from src.api.routes.health import router as health_router
from src.api.routes.forecasting import router as forecasting_router
from src.api.routes.hierarchy import router as hierarchy_router
from src.api.routes.inventory import router as inventory_router
from src.api.routes.monitoring import router as monitoring_router
from src.api.routes.experiments import router as experiments_router

__all__ = [
    "health_router",
    "forecasting_router",
    "hierarchy_router",
    "inventory_router",
    "monitoring_router",
    "experiments_router",
]

