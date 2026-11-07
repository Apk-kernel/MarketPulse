"""Comprehensive tests for MarketPulse API routes, schemas, and services."""

import pytest
from src.api.main import app
from src.api.services import MarketPulseService
from src.api.schemas import (
    HealthResponse,
    MetadataResponse,
    ForecastRequest,
    ForecastResponse,
    ReconcileRequest,
    ReconcileResponse,
    InventoryPolicyRequest,
    InventoryPolicyResponse,
    InventorySimulationRequest,
    InventorySimulationResponse,
    DriftAnalysisResponse,
    RetrainingDecisionResponse,
    BenchmarkResultsResponse,
)
from src.api.routes.health import health_check, get_system_metadata
from src.api.routes.forecasting import generate_forecast, list_available_models
from src.api.routes.hierarchy import reconcile_forecasts, get_hierarchy_levels
from src.api.routes.inventory import optimize_inventory_policy, simulate_inventory
from src.api.routes.monitoring import get_drift_analysis, get_retraining_decision
from src.api.routes.experiments import get_benchmark_results


@pytest.fixture(scope="module")
def api_service():
    """Provides cached MarketPulseService instance."""
    return MarketPulseService()


def test_api_route_registration():
    """Validates that all API v1 endpoints are registered on the FastAPI app."""
    route_paths = [r.path for r in app.routes]
    expected_endpoints = [
        "/api/v1/health",
        "/api/v1/metadata",
        "/api/v1/forecast",
        "/api/v1/models",
        "/api/v1/hierarchy/reconcile",
        "/api/v1/hierarchy/levels",
        "/api/v1/inventory/policy",
        "/api/v1/inventory/simulate",
        "/api/v1/monitoring/drift",
        "/api/v1/monitoring/retraining-decision",
        "/api/v1/experiments/benchmarks",
    ]
    for endpoint in expected_endpoints:
        assert endpoint in route_paths, f"Missing route: {endpoint}"


def test_health_check_endpoint():
    """Validates health check route handler returns healthy status and timestamp."""
    resp = health_check()
    assert isinstance(resp, HealthResponse)
    assert resp.status == "healthy"
    assert resp.version == "0.1.0"
    assert len(resp.timestamp) > 0


def test_metadata_endpoint(api_service):
    """Validates system metadata route handler returns 12 hierarchy levels."""
    resp = get_system_metadata(service=api_service)
    assert isinstance(resp, MetadataResponse)
    assert resp.project_name == "MarketPulse"
    assert resp.n_series > 0
    assert resp.n_days > 0
    assert len(resp.available_series) > 0
    assert len(resp.hierarchy_levels) == 12
    assert resp.hierarchy_levels[0].level == 1
    assert resp.hierarchy_levels[11].level == 12


def test_forecast_endpoint_lightgbm(api_service):
    """Validates demand forecasting route handler with recursive LightGBM."""
    req = ForecastRequest(
        series_id=api_service.get_data()["id"].iloc[0],
        horizon=14,
        model_name="lightgbm",
        quantiles=[0.05, 0.50, 0.95],
    )
    resp = generate_forecast(request=req, service=api_service)
    assert isinstance(resp, ForecastResponse)
    assert resp.horizon == 14
    assert len(resp.point_forecast) == 14
    assert all(v >= 0.0 for v in resp.point_forecast)
    assert len(resp.historical_actuals) > 0
    assert len(resp.forecast_dates) == 14
    assert "q_05" in resp.quantiles
    assert "q_50" in resp.quantiles
    assert "q_95" in resp.quantiles
    assert "WAPE" in resp.metrics
    assert "MAE" in resp.metrics


def test_forecast_endpoint_seasonal_naive(api_service):
    """Validates demand forecasting route handler with Seasonal Naive."""
    req = ForecastRequest(
        series_id=api_service.get_data()["id"].iloc[0],
        horizon=7,
        model_name="seasonal_naive",
    )
    resp = generate_forecast(request=req, service=api_service)
    assert isinstance(resp, ForecastResponse)
    assert resp.horizon == 7
    assert len(resp.point_forecast) == 7
    assert all(v >= 0.0 for v in resp.point_forecast)


def test_available_models_endpoint():
    """Validates list_available_models route handler."""
    models = list_available_models()
    assert "lightgbm" in models
    assert "auto_arima" in models
    assert "croston" in models
    assert "seasonal_naive" in models


def test_hierarchy_reconciliation_endpoint(api_service):
    """Validates hierarchical reconciliation route handler enforces coherence."""
    req = ReconcileRequest(method="mint_shrink", horizon=7)
    resp = reconcile_forecasts(request=req, service=api_service)
    assert isinstance(resp, ReconcileResponse)
    assert resp.method == "mint_shrink"
    assert resp.horizon == 7
    assert resp.coherence_verified is True
    assert resp.max_coherence_error < 1e-4
    assert len(resp.total_sales_reconciled) == 7
    assert len(resp.sample_reconciled_series) > 0


def test_hierarchy_levels_endpoint(api_service):
    """Validates get_hierarchy_levels route handler."""
    levels = get_hierarchy_levels(service=api_service)
    assert len(levels) == 12
    assert levels[0].name == "Total"


def test_inventory_policy_endpoint(api_service):
    """Validates inventory policy calculation route handler."""
    req = InventoryPolicyRequest(
        daily_demand_mean=12.0,
        daily_demand_std=3.5,
        lead_time_mean_days=4.0,
        lead_time_std_days=1.2,
        unit_cost=8.0,
        holding_cost_rate=0.20,
        fixed_order_cost=60.0,
        target_csl=0.95,
    )
    resp = optimize_inventory_policy(request=req, service=api_service)
    assert isinstance(resp, InventoryPolicyResponse)
    assert resp.safety_stock > 0.0
    assert resp.reorder_point > resp.safety_stock
    assert resp.economic_order_quantity > 0.0
    assert resp.target_stock_level > resp.reorder_point
    assert resp.total_annual_cost > 0.0


def test_inventory_simulation_endpoint(api_service):
    """Validates discrete-event inventory simulation route handler."""
    req = InventorySimulationRequest(
        series_id=api_service.get_data()["id"].iloc[0],
        simulation_days=30,
        seed=42,
    )
    resp = simulate_inventory(request=req, service=api_service)
    assert isinstance(resp, InventorySimulationResponse)
    assert resp.simulation_days == 30
    kpis = resp.kpis
    assert 0.0 <= kpis["fill_rate"] <= 1.0
    assert 0.0 <= kpis["cycle_service_level"] <= 1.0
    assert kpis["total_cost_of_inventory"] > 0.0
    assert len(resp.timeline.days) == 30
    assert len(resp.timeline.demand) == 30
    assert len(resp.timeline.sales) == 30
    assert len(resp.timeline.on_hand) == 30


def test_monitoring_drift_endpoint(api_service):
    """Validates feature drift analysis route handler."""
    resp = get_drift_analysis(window="Window_1", service=api_service)
    assert isinstance(resp, DriftAnalysisResponse)
    assert len(resp.features) > 0
    assert resp.overall_drift_status in ["STABLE", "MODERATE_DRIFT", "SEVERE_DRIFT"]
    assert "target_drift" in resp.model_dump()


def test_monitoring_retraining_decisions(api_service):
    """Validates retraining decision route handler for both windows."""
    # Window 1: Stable Operations -> NO_ACTION
    resp_w1 = get_retraining_decision(window="window_1", service=api_service)
    assert isinstance(resp_w1, RetrainingDecisionResponse)
    assert resp_w1.decision == "NO_ACTION"
    assert resp_w1.is_retraining_justified is False
    assert resp_w1.urgency == "LOW"

    # Window 2: Demand Shock -> CRITICAL_FALLBACK
    resp_w2 = get_retraining_decision(window="window_2", service=api_service)
    assert isinstance(resp_w2, RetrainingDecisionResponse)
    assert resp_w2.decision == "CRITICAL_FALLBACK"
    assert resp_w2.is_retraining_justified is True
    assert resp_w2.urgency == "CRITICAL"


def test_research_benchmarks_endpoint(api_service):
    """Validates research experiment benchmarks route handler."""
    resp = get_benchmark_results(service=api_service)
    assert isinstance(resp, BenchmarkResultsResponse)
    assert resp.total_experiments == 6
    exp_ids = [e.experiment_id for e in resp.experiments]
    assert exp_ids == ["EXP-1", "EXP-2", "EXP-3", "EXP-4", "EXP-5", "EXP-6"]


def test_root_serves_frontend():
    """Validates that root route serves the compiled frontend index.html."""
    from src.api.main import root
    from fastapi.responses import FileResponse
    
    resp = root()
    assert isinstance(resp, FileResponse)
    assert str(resp.path).endswith("index.html")
    assert "frontend" in str(resp.path)



