"""MarketPulse FastAPI Application Entrypoint."""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from src.utils.logging import setup_logger
from src.api.routes import (
    health_router,
    forecasting_router,
    hierarchy_router,
    inventory_router,
    monitoring_router,
    experiments_router,
)

logger = setup_logger("MarketPulse.API")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application startup and shutdown events."""
    logger.info("Initializing MarketPulse API services...")
    # Pre-warm service data cache
    from src.api.services import MarketPulseService
    try:
        service = MarketPulseService()
        df = service.get_data()
        logger.info(f"MarketPulse dataset cached: {len(df)} rows across {df['id'].nunique()} series.")
    except Exception as exc:
        logger.warning(f"Pre-warm skipped (dataset may not be built yet): {exc}")
    yield
    logger.info("MarketPulse API service shutting down.")


app = FastAPI(
    title="MarketPulse Enterprise Demand Forecasting & Decision Engine",
    description=(
        "Production RESTful API serving hierarchical probabilistic demand forecasts, "
        "coherent reconciliation, Monte Carlo inventory replenishment policies, "
        "and automated drift detection / model retraining governance."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for frontend dashboard development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Routers
API_V1_PREFIX = "/api/v1"
app.include_router(health_router, prefix=API_V1_PREFIX)
app.include_router(forecasting_router, prefix=API_V1_PREFIX)
app.include_router(hierarchy_router, prefix=API_V1_PREFIX)
app.include_router(inventory_router, prefix=API_V1_PREFIX)
app.include_router(monitoring_router, prefix=API_V1_PREFIX)
app.include_router(experiments_router, prefix=API_V1_PREFIX)


@app.get("/", include_in_schema=False)
def root():
    """Redirects root URL to API documentation or frontend dashboard."""
    project_root = Path(__file__).resolve().parent.parent.parent
    dist_index = project_root / "frontend" / "dist" / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)
    return RedirectResponse(url="/docs")


# Mount static assets if built frontend is present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIST.exists():
    logger.info(f"Mounting production frontend build from {FRONTEND_DIST}")
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        """Catches SPA client-side routes and serves index.html."""
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
