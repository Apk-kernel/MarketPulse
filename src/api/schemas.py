"""Pydantic request and response schemas for the MarketPulse FastAPI backend."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & System Metadata
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field("healthy", description="System operational status")
    version: str = Field("0.1.0", description="MarketPulse API version")
    timestamp: str = Field(..., description="ISO 8601 server timestamp")


class HierarchyLevelInfo(BaseModel):
    """Schema representing an aggregation level in the hierarchy."""
    level: int = Field(..., description="M5 hierarchy level (1 to 12)")
    name: str = Field(..., description="Level display name")
    columns: List[str] = Field(..., description="Grouping columns")
    num_nodes: int = Field(..., description="Number of unique series/nodes in level")


class MetadataResponse(BaseModel):
    """System metadata response containing hierarchy layout and dataset dimensions."""
    project_name: str
    version: str
    n_series: int
    n_days: int
    start_date: str
    end_date: str
    available_series: List[str]
    states: List[str]
    stores: List[str]
    categories: List[str]
    departments: List[str]
    hierarchy_levels: List[HierarchyLevelInfo]
    available_models: List[str]


# ---------------------------------------------------------------------------
# Demand Forecasting
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    """Demand forecast generation request parameters."""
    series_id: Optional[str] = Field("CA_1_FOODS_1_001", description="Specific SKU-Store time series ID")
    hierarchy_level: int = Field(12, ge=1, le=12, description="Hierarchy level (12 = bottom SKU-store)")
    aggregation_key: Optional[str] = Field(None, description="Optional key if level < 12 (e.g. 'CA' or 'FOODS')")
    horizon: int = Field(28, ge=1, le=56, description="Forecast horizon in days")
    model_name: str = Field("lightgbm", description="Forecaster: 'lightgbm', 'auto_arima', 'croston', 'seasonal_naive'")
    quantiles: List[float] = Field(
        default=[0.05, 0.10, 0.50, 0.90, 0.95],
        description="Target prediction quantiles for interval forecasting",
    )


class ForecastResponse(BaseModel):
    """Demand forecast response with point predictions and probabilistic fan intervals."""
    series_id: str
    hierarchy_level: int
    model_name: str
    horizon: int
    historical_dates: List[str]
    historical_actuals: List[float]
    forecast_dates: List[str]
    point_forecast: List[float]
    quantiles: Dict[str, List[float]]
    metrics: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hierarchical Reconciliation
# ---------------------------------------------------------------------------
class ReconcileRequest(BaseModel):
    """Hierarchical reconciliation request."""
    method: str = Field("mint_shrink", description="Reconciliation method: 'mint_shrink', 'bottom_up', 'ols'")
    horizon: int = Field(14, ge=1, le=28, description="Reconciliation forecast horizon")
    sample_series_ids: Optional[List[str]] = Field(None, description="Subset of series to reconcile")


class ReconcileLevelSeries(BaseModel):
    """Individual series reconciliation outcome."""
    name: str
    base_forecast: List[float]
    reconciled_forecast: List[float]
    adjustment_delta: List[float]


class ReconcileResponse(BaseModel):
    """Coherent multi-level hierarchical reconciliation results."""
    method: str
    horizon: int
    forecast_dates: List[str]
    coherence_verified: bool
    max_coherence_error: float
    total_sales_base: List[float]
    total_sales_reconciled: List[float]
    level_aggregates: Dict[str, Dict[str, float]]
    sample_reconciled_series: List[ReconcileLevelSeries]


# ---------------------------------------------------------------------------
# Inventory Decision Engine & Simulation
# ---------------------------------------------------------------------------
class InventoryPolicyRequest(BaseModel):
    """Parameters for analytical replenishment policy optimization."""
    daily_demand_mean: float = Field(10.0, gt=0, description="Mean daily customer demand")
    daily_demand_std: float = Field(3.0, ge=0, description="Daily demand standard deviation")
    lead_time_mean_days: float = Field(3.0, gt=0, description="Supplier mean replenishment lead time in days")
    lead_time_std_days: float = Field(1.0, ge=0, description="Replenishment lead time standard deviation")
    unit_cost: float = Field(5.0, gt=0, description="Unit acquisition cost ($)")
    holding_cost_rate: float = Field(0.25, gt=0, description="Annual holding cost rate (% of unit cost)")
    fixed_order_cost: float = Field(50.0, gt=0, description="Fixed order placement cost ($)")
    target_csl: float = Field(0.95, gt=0.5, lt=1.0, description="Target Cycle Service Level (e.g. 0.95 for 95%)")
    review_period_days: int = Field(1, ge=1, description="Periodic review interval in days")


class InventoryPolicyResponse(BaseModel):
    """Calculated replenishment parameters and annual cost breakdown."""
    safety_stock: float
    reorder_point: float
    economic_order_quantity: float
    target_stock_level: float
    lead_time_demand_mean: float
    lead_time_demand_std: float
    annual_holding_cost: float
    annual_ordering_cost: float
    total_annual_cost: float
    recommended_policy: str


class InventorySimulationRequest(BaseModel):
    """Discrete-event Monte Carlo inventory simulation request."""
    series_id: Optional[str] = Field("CA_1_FOODS_1_001", description="SKU series for real demand simulation")
    policy_params: Optional[InventoryPolicyRequest] = None
    simulation_days: int = Field(60, ge=14, le=90, description="Simulation duration in days")
    initial_on_hand: Optional[float] = Field(None, description="Initial starting on-hand inventory")
    seed: int = Field(42, description="Random seed for lead-time stochasticity")


class SimulationTimeline(BaseModel):
    """Daily time series trajectory of inventory states."""
    days: List[int]
    dates: List[str]
    demand: List[float]
    sales: List[float]
    lost_sales: List[float]
    on_hand: List[float]
    on_order: List[float]
    inventory_position: List[float]
    order_placed_qty: List[float]


class InventorySimulationResponse(BaseModel):
    """Monte Carlo simulation outcome and supply chain performance KPIs."""
    simulation_days: int
    policy: Dict[str, float]
    kpis: Dict[str, float]
    timeline: SimulationTimeline


# ---------------------------------------------------------------------------
# Production Monitoring & Drift Detection
# ---------------------------------------------------------------------------
class DriftFeatureResult(BaseModel):
    """Individual feature drift metrics."""
    feature_name: str
    feature_type: str
    psi: float
    psi_status: str
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    ks_drift_detected: Optional[bool] = None
    wasserstein_dist: Optional[float] = None


class DriftAnalysisResponse(BaseModel):
    """Covariate and target drift evaluation summary."""
    monitoring_window: str
    reference_sample_size: int
    current_sample_size: int
    features: List[DriftFeatureResult]
    max_psi: float
    overall_drift_status: str
    target_drift: Dict[str, Any]


class RetrainingDecisionResponse(BaseModel):
    """Objective 4-tier model retraining governance decision."""
    monitoring_window: str
    model_name: str
    decision: str
    is_retraining_justified: bool
    urgency: str
    primary_reason: str
    detailed_justification: List[str]
    recommended_action: str
    baseline_wape: float
    current_wape: float
    delta_wape_pct: float
    tracking_signal: float
    bias_z_score: float
    cusum_alarm: bool


# ---------------------------------------------------------------------------
# Research Experiments & Benchmark Results
# ---------------------------------------------------------------------------
class BenchmarkExperimentSummary(BaseModel):
    """Research experiment benchmark summary record."""
    experiment_id: str
    title: str
    description: str
    key_finding: str
    metrics_summary: Dict[str, Any]


class BenchmarkResultsResponse(BaseModel):
    """Aggregated research experiments benchmark explorer response."""
    total_experiments: int
    experiments: List[BenchmarkExperimentSummary]

