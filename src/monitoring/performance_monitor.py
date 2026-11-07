"""Model Performance, Residual Tracking, and Retraining Recommendation Engine.

Monitors live forecasting accuracy, tracks residual distributions, and detects concept drift:
1. Multi-metric temporal tracking (WAPE, MAE, RMSE, normalized bias).
2. Tracking Signal (TS = CFE / MAD) for persistent systematic bias detection.
3. Two-sided CUSUM (Cumulative Sum Control Chart) for residual mean-shift detection.
4. Objective Retraining Decision Matrix: Evaluates whether retraining is empirically justified
   or whether performance variations are within acceptable statistical tolerance.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import (
    calculate_forecast_metrics,
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Monitoring.PerformanceMonitor")


class RetrainingStatus(str, Enum):
    """Operational status of model retraining recommendation."""

    NO_ACTION = "NO_ACTION"
    WARNING = "WARNING"
    RETRAIN_RECOMMENDED = "RETRAIN_RECOMMENDED"
    CRITICAL_FALLBACK = "CRITICAL_FALLBACK"


@dataclass
class RetrainingDecision:
    """Actionable recommendation produced by the retraining evaluation engine."""

    decision: RetrainingStatus
    is_retraining_justified: bool
    urgency: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    primary_reason: str
    detailed_justification: List[str]
    recommended_action: str
    baseline_wape: float
    current_wape: float
    delta_wape_pct: float
    max_tracking_signal: float
    latest_tracking_signal: float
    bias_z_score: float
    cusum_alarm_triggered: bool
    max_feature_psi: Optional[float] = None
    target_psi: Optional[float] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


def calculate_tracking_signal(
    actuals: np.ndarray,
    predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Cumulative Forecast Error (CFE), Mean Absolute Deviation (MAD), and Tracking Signal (TS).

    Formulas:
        CFE_t = sum_{i=1}^t (y_i - y_hat_i)
        MAD_t = (1 / t) * sum_{i=1}^t |y_i - y_hat_i|
        TS_t  = CFE_t / (MAD_t + eps)

    Control Limits:
        -4.0 <= TS <= +4.0 is typical industrial standard (~3.75 standard deviations).
        Values outside [-4.0, +4.0] indicate significant persistent forecast bias.

    Args:
        actuals: Ground truth observed demand.
        predictions: Model point forecasts.

    Returns:
        Tuple of (cfe_array, mad_array, tracking_signal_array).
    """
    acts = np.asarray(actuals, dtype=float)
    preds = np.asarray(predictions, dtype=float)
    n = len(acts)

    if n == 0:
        return np.array([]), np.array([]), np.array([])

    errors = acts - preds
    abs_errors = np.abs(errors)

    cfe = np.cumsum(errors)
    t_steps = np.arange(1, n + 1, dtype=float)
    mad = np.cumsum(abs_errors) / t_steps

    eps = 1e-6
    ts = cfe / (mad + eps)

    return np.round(cfe, 3), np.round(mad, 3), np.round(ts, 3)


def cusum_drift_detector(
    residuals: np.ndarray,
    slack_k: float = 0.5,
    threshold_h: float = 4.0,
    baseline_mean: Optional[float] = None,
    baseline_std: Optional[float] = None,
) -> Dict[str, Any]:
    """Two-sided Tabular Cumulative Sum (CUSUM) control chart for residual mean shifts.

    Formulas:
        z_t = (e_t - mu_0) / sigma_0
        S_t^+ = max(0, S_{t-1}^+ + z_t - k)
        S_t^- = max(0, S_{t-1}^- - z_t - k)
        Alarm if max(S_t^+, S_t^-) > h

    Args:
        residuals: Time-ordered forecast errors (actual - prediction).
        slack_k: Slack allowance parameter (default: 0.5 for detecting a 1-sigma shift).
        threshold_h: Decision interval threshold (default: 4.0 to 5.0).
        baseline_mean: Reference baseline mean residual (default: None, estimated from sample).
        baseline_std: Reference baseline residual standard deviation.

    Returns:
        Dictionary with positive/negative CUSUM series, alarm flag, and first change point index.
    """
    res_clean = np.asarray(residuals, dtype=float)
    res_clean = res_clean[np.isfinite(res_clean)]
    n = len(res_clean)

    if n < 5:
        return {
            "alarm": False,
            "change_point_idx": None,
            "max_s_pos": 0.0,
            "max_s_neg": 0.0,
            "s_pos": [],
            "s_neg": [],
        }

    if baseline_mean is not None:
        mu = float(baseline_mean)
    else:
        # Default in-control baseline mean for forecast residuals is zero (unbiased model)
        mu = 0.0

    if baseline_std is not None and baseline_std > 0:
        sigma = float(baseline_std)
    else:
        sigma = max(float(np.std(res_clean)), 1e-4)

    z = (res_clean - mu) / sigma

    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    alarm = False
    change_point_idx = None

    for t in range(1, n):
        s_pos[t] = max(0.0, s_pos[t - 1] + z[t] - slack_k)
        s_neg[t] = max(0.0, s_neg[t - 1] - z[t] - slack_k)

        if not alarm and (s_pos[t] > threshold_h or s_neg[t] > threshold_h):
            alarm = True
            change_point_idx = t

    return {
        "alarm": alarm,
        "change_point_idx": change_point_idx,
        "max_s_pos": round(float(np.max(s_pos)), 3),
        "max_s_neg": round(float(np.max(s_neg)), 3),
        "s_pos": [round(float(v), 3) for v in s_pos],
        "s_neg": [round(float(v), 3) for v in s_neg],
    }


class ModelPerformanceMonitor:
    """Tracks live model forecasting accuracy, residual distributions, and retraining necessity."""

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        model_name: str = "Champion_Model",
        sp: int = 7,
    ):
        """Initializes monitor with historical baseline reference performance.

        Args:
            baseline_metrics: Dictionary of benchmark validation metrics (must contain 'WAPE').
            model_name: Identifier of monitored model.
            sp: Seasonal periodicity for metric scaling (default: 7).
        """
        self.baseline_metrics = baseline_metrics
        self.baseline_wape = float(baseline_metrics.get("WAPE", 0.0))
        self.model_name = model_name
        self.sp = sp
        self.history: List[Dict[str, Any]] = []

    def record_evaluation_window(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        window_id: str,
        dates: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluates a live monitoring time window, computing accuracy metrics and tracking signals.

        Args:
            actuals: Observed actual demand.
            predictions: Forecasted demand.
            window_id: Window identifier (e.g. '2021-W15').
            dates: Optional list of timestamps.

        Returns:
            Dictionary of window evaluation results.
        """
        acts = np.asarray(actuals, dtype=float)
        preds = np.asarray(predictions, dtype=float)

        wape = weighted_absolute_percentage_error(acts, preds)
        mae = mean_absolute_error(acts, preds)
        rmse = root_mean_squared_error(acts, preds)
        residuals = acts - preds

        if dates is not None and len(dates) == len(acts):
            # Aggregate portfolio daily totals for sequential temporal control charts
            daily_df = (
                pd.DataFrame({"date": dates, "actual": acts, "pred": preds})
                .groupby("date")[["actual", "pred"]]
                .sum()
                .sort_index()
            )
            temporal_acts = daily_df["actual"].values
            temporal_preds = daily_df["pred"].values
            temporal_res = temporal_acts - temporal_preds
        else:
            temporal_acts = acts
            temporal_preds = preds
            temporal_res = residuals

        cfe, mad, ts = calculate_tracking_signal(temporal_acts, temporal_preds)
        cusum_res = cusum_drift_detector(temporal_res)

        delta_wape = ((wape - self.baseline_wape) / (self.baseline_wape + 1e-4)) * 100.0

        record = {
            "window_id": window_id,
            "n_obs": len(acts),
            "WAPE": round(wape, 2),
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "delta_wape_pct": round(delta_wape, 2),
            "mean_residual": round(float(np.mean(residuals)), 3),
            "std_residual": round(float(np.std(residuals)), 3),
            "max_tracking_signal": round(float(np.max(np.abs(ts))), 2) if len(ts) > 0 else 0.0,
            "latest_tracking_signal": round(float(ts[-1]), 2) if len(ts) > 0 else 0.0,
            "cusum_alarm": cusum_res["alarm"],
            "cusum_change_point": cusum_res["change_point_idx"],
        }
        self.history.append(record)
        return record

    def evaluate_retraining_necessity(
        self,
        current_actuals: np.ndarray,
        current_predictions: np.ndarray,
        dates: Optional[List[Any]] = None,
        max_feature_psi: Optional[float] = None,
        target_psi: Optional[float] = None,
        significance_threshold_pct: float = 15.0,
        critical_threshold_pct: float = 50.0,
    ) -> RetrainingDecision:
        """Determines whether model retraining is statistically and economically justified.

        Follows strict scientific justification guidelines:
        - Avoids unnecessary retraining when errors are within stochastic noise.
        - Recommends retraining when:
          1. Significant error degradation (Delta WAPE > 15%) is observed.
          2. Persistent forecast bias (|Tracking Signal| > 4.0) or CUSUM alarm is detected.
          3. Severe covariate shift (PSI >= 0.25) aligns with deteriorating performance.
        - Triggers Critical Fallback if error degradation exceeds 50%.

        Args:
            current_actuals: Ground truth observed demand in monitoring window.
            current_predictions: Live model predictions.
            dates: Optional time indices (dates) for daily portfolio temporal aggregation.
            max_feature_psi: Maximum observed PSI across exogenous features.
            target_psi: Observed PSI on the target sales distribution.
            significance_threshold_pct: WAPE increase threshold for recommendation (default: 15%).
            critical_threshold_pct: WAPE increase threshold for critical fallback (default: 50%).

        Returns:
            RetrainingDecision instance.
        """
        acts = np.asarray(current_actuals, dtype=float)
        preds = np.asarray(current_predictions, dtype=float)
        n = len(acts)

        wape_curr = weighted_absolute_percentage_error(acts, preds)
        delta_wape = ((wape_curr - self.baseline_wape) / (self.baseline_wape + 1e-4)) * 100.0

        residuals = acts - preds
        mean_err = float(np.mean(residuals))
        std_err = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        z_bias = mean_err / (std_err / np.sqrt(n) + 1e-6)

        if dates is not None and len(dates) == len(acts):
            # Aggregate portfolio daily totals for sequential temporal control charts
            daily_df = (
                pd.DataFrame({"date": dates, "actual": acts, "pred": preds})
                .groupby("date")[["actual", "pred"]]
                .sum()
                .sort_index()
            )
            temporal_acts = daily_df["actual"].values
            temporal_preds = daily_df["pred"].values
            temporal_res = temporal_acts - temporal_preds
        else:
            temporal_acts = acts
            temporal_preds = preds
            temporal_res = residuals

        cfe, mad, ts = calculate_tracking_signal(temporal_acts, temporal_preds)
        latest_ts = float(ts[-1]) if len(ts) > 0 else 0.0
        max_ts = float(np.max(np.abs(ts))) if len(ts) > 0 else 0.0
        cusum_res = cusum_drift_detector(temporal_res)
        cusum_alarm = cusum_res["alarm"]

        justifications = []
        decision = RetrainingStatus.NO_ACTION
        urgency = "LOW"
        is_justified = False
        primary_reason = "Model performance is stable within statistical tolerance."
        recommended_action = "Maintain current model in production; no retraining required."

        # Diagnostic flags
        severe_error_degradation = delta_wape >= significance_threshold_pct
        critical_error_degradation = delta_wape >= critical_threshold_pct
        # Persistent systematic bias: confirmed when both tracking signal is high and z_bias is statistically significant
        persistent_bias = (abs(z_bias) >= 3.0) and (abs(latest_ts) >= 3.5)
        covariate_drift = max_feature_psi is not None and max_feature_psi >= 0.25
        target_drift = target_psi is not None and target_psi >= 0.25

        if critical_error_degradation:
            decision = RetrainingStatus.CRITICAL_FALLBACK
            urgency = "CRITICAL"
            is_justified = True
            primary_reason = f"Catastrophic error degradation: WAPE increased by {round(delta_wape, 1)}%."
            justifications.append(
                f"Current WAPE ({round(wape_curr, 2)}%) exceeds critical threshold (+{critical_threshold_pct}% over baseline {round(self.baseline_wape, 2)}%)."
            )
            if persistent_bias:
                justifications.append(
                    f"Severe persistent systematic bias: |Tracking Signal| = {round(abs(latest_ts), 2)} > 3.5 (Z = {round(z_bias, 2)})."
                )
            recommended_action = (
                "IMMEDIATE ACTION: Switch serving traffic to robust fallback baseline (Seasonal Naive / Croston) "
                "and trigger immediate high-priority model retraining and hyperparameter re-tuning."
            )

        elif severe_error_degradation or persistent_bias or cusum_alarm:
            decision = RetrainingStatus.RETRAIN_RECOMMENDED
            urgency = "HIGH"
            is_justified = True
            reasons = []
            if severe_error_degradation:
                reasons.append(f"WAPE deteriorated by {round(delta_wape, 1)}% (>{significance_threshold_pct}%)")
                justifications.append(
                    f"Operational error ({round(wape_curr, 2)}% WAPE) significantly worse than baseline ({round(self.baseline_wape, 2)}% WAPE)."
                )
            if persistent_bias:
                reasons.append(f"persistent bias (|TS| = {round(abs(latest_ts), 2)}, Z = {round(z_bias, 2)})")
                justifications.append(
                    f"Forecast exhibits non-zero systematic error accumulation: Tracking Signal {round(latest_ts, 2)} exceeds control limit with Z = {round(z_bias, 2)}."
                )
            if cusum_alarm:
                reasons.append(f"residual mean shift detected by CUSUM at step {cusum_res['change_point_idx']}")
                justifications.append(
                    f"CUSUM control chart detected structural residual shift (max S+={cusum_res['max_s_pos']}, S-={cusum_res['max_s_neg']})."
                )
            if covariate_drift:
                justifications.append(f"Corroborated by severe feature covariate shift (max PSI = {round(max_feature_psi, 3)} >= 0.25).")

            primary_reason = f"Statistically justified retraining trigger: {', '.join(reasons)}."
            recommended_action = (
                "Schedule model retraining pipeline with recent historical data and update baseline calibration weights."
            )

        elif (delta_wape > 5.0) or (max_feature_psi is not None and max_feature_psi >= 0.10) or target_drift:
            decision = RetrainingStatus.WARNING
            urgency = "MEDIUM"
            is_justified = False
            reasons = []
            if delta_wape > 5.0:
                reasons.append(f"mild accuracy degradation (+{round(delta_wape, 1)}% WAPE)")
            if max_feature_psi is not None and max_feature_psi >= 0.10:
                reasons.append(f"moderate feature drift (PSI = {round(max_feature_psi, 3)})")
            if target_drift:
                reasons.append(f"target distribution drift (PSI = {round(target_psi, 3)})")

            primary_reason = f"Warning condition detected: {', '.join(reasons)}, but retraining is NOT yet justified."
            justifications.append(
                "Error increase and drift metrics remain within acceptable operational control limits. "
                "Retraining now would introduce unnecessary compute cost and variance."
            )
            recommended_action = (
                "Continue monitoring live traffic with heightened sampling frequency; do NOT retrain yet."
            )

        else:
            justifications.append(
                f"Operational WAPE ({round(wape_curr, 2)}%) is consistent with baseline ({round(self.baseline_wape, 2)}%)."
            )
            justifications.append(f"Residual tracking signal (|TS| = {round(abs(latest_ts), 2)}) is well within standard limits (<= 3.5).")
            if max_feature_psi is not None:
                justifications.append(f"Feature stability verified: Max PSI = {round(max_feature_psi, 3)} < 0.10.")

        return RetrainingDecision(
            decision=decision,
            is_retraining_justified=is_justified,
            urgency=urgency,
            primary_reason=primary_reason,
            detailed_justification=justifications,
            recommended_action=recommended_action,
            baseline_wape=round(self.baseline_wape, 2),
            current_wape=round(wape_curr, 2),
            delta_wape_pct=round(delta_wape, 2),
            max_tracking_signal=round(max_ts, 2),
            latest_tracking_signal=round(latest_ts, 2),
            bias_z_score=round(z_bias, 3),
            cusum_alarm_triggered=cusum_alarm,
            max_feature_psi=round(max_feature_psi, 3) if max_feature_psi is not None else None,
            target_psi=round(target_psi, 3) if target_psi is not None else None,
        )
