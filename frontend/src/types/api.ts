/**
 * MarketPulse API Types & Data Contracts
 */

export interface HierarchyLevelInfo {
  level: number;
  name: string;
  columns: string[];
  num_nodes: number;
}

export interface MetadataResponse {
  project_name: string;
  version: string;
  n_series: number;
  n_days: number;
  start_date: string;
  end_date: string;
  available_series: string[];
  states: string[];
  stores: string[];
  categories: string[];
  departments: string[];
  hierarchy_levels: HierarchyLevelInfo[];
  available_models: string[];
}

export interface ForecastRequest {
  series_id: string;
  hierarchy_level?: number;
  aggregation_key?: string;
  horizon: number;
  model_name: string;
  quantiles?: number[];
}

export interface ForecastResponse {
  series_id: string;
  hierarchy_level: number;
  model_name: string;
  horizon: number;
  historical_dates: string[];
  historical_actuals: number[];
  forecast_dates: string[];
  point_forecast: number[];
  quantiles: Record<string, number[]>;
  metrics: Record<string, number>;
}

export interface ReconcileRequest {
  method: string;
  horizon: number;
  sample_series_ids?: string[];
}

export interface ReconcileLevelSeries {
  name: string;
  base_forecast: number[];
  reconciled_forecast: number[];
  adjustment_delta: number[];
}

export interface ReconcileResponse {
  method: string;
  horizon: number;
  forecast_dates: string[];
  coherence_verified: boolean;
  max_coherence_error: number;
  total_sales_base: number[];
  total_sales_reconciled: number[];
  level_aggregates: Record<string, Record<string, number>>;
  sample_reconciled_series: ReconcileLevelSeries[];
}

export interface InventoryPolicyRequest {
  daily_demand_mean: number;
  daily_demand_std: number;
  lead_time_mean_days: number;
  lead_time_std_days: number;
  unit_cost: number;
  holding_cost_rate: number;
  fixed_order_cost: number;
  target_csl: number;
  review_period_days?: number;
}

export interface InventoryPolicyResponse {
  safety_stock: number;
  reorder_point: number;
  economic_order_quantity: number;
  target_stock_level: number;
  lead_time_demand_mean: number;
  lead_time_demand_std: number;
  annual_holding_cost: number;
  annual_ordering_cost: number;
  total_annual_cost: number;
  recommended_policy: string;
}

export interface InventorySimulationRequest {
  series_id: string;
  policy_params?: InventoryPolicyRequest;
  simulation_days?: number;
  initial_on_hand?: number;
  seed?: number;
}

export interface SimulationTimeline {
  days: number[];
  dates: string[];
  demand: number[];
  sales: number[];
  lost_sales: number[];
  on_hand: number[];
  on_order: number[];
  inventory_position: number[];
  order_placed_qty: number[];
}

export interface InventorySimulationResponse {
  simulation_days: number;
  policy: Record<string, number>;
  kpis: Record<string, number>;
  timeline: SimulationTimeline;
}

export interface DriftFeatureResult {
  feature_name: string;
  feature_type: string;
  psi: number;
  psi_status: string;
  ks_statistic?: number;
  ks_p_value?: number;
  ks_drift_detected?: boolean;
  wasserstein_dist?: number;
}

export interface DriftAnalysisResponse {
  monitoring_window: string;
  reference_sample_size: number;
  current_sample_size: number;
  features: DriftFeatureResult[];
  max_psi: number;
  overall_drift_status: string;
  target_drift: Record<string, any>;
}

export interface RetrainingDecisionResponse {
  monitoring_window: string;
  model_name: string;
  decision: string;
  is_retraining_justified: boolean;
  urgency: string;
  primary_reason: string;
  detailed_justification: string[];
  recommended_action: string;
  baseline_wape: number;
  current_wape: number;
  delta_wape_pct: number;
  tracking_signal: number;
  bias_z_score: number;
  cusum_alarm: boolean;
}

export interface BenchmarkExperimentSummary {
  experiment_id: string;
  title: string;
  description: string;
  key_finding: string;
  metrics_summary: Record<string, any>;
}

export interface BenchmarkResultsResponse {
  total_experiments: number;
  experiments: BenchmarkExperimentSummary[];
}

