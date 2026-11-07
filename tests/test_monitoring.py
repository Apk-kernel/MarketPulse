"""Comprehensive unit and integration test suite for Phase 11: Production Monitoring & Drift Detection."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detection import (
    DataDriftDetector,
    calculate_psi,
    classify_psi,
    kolmogorov_smirnov_test,
    wasserstein_distance_test,
    categorical_divergence_test,
)
from src.monitoring.performance_monitor import (
    ModelPerformanceMonitor,
    RetrainingDecision,
    RetrainingStatus,
    calculate_tracking_signal,
    cusum_drift_detector,
)


def test_psi_identical_and_shifted_distributions():
    """Validates Population Stability Index: near-zero for identical distributions, > 0.25 for shifted."""
    rng = np.random.default_rng(42)
    ref = rng.normal(10.0, 2.0, size=2000)
    curr_identical = rng.normal(10.0, 2.0, size=2000)
    curr_shifted = rng.normal(13.0, 2.0, size=2000)  # 1.5 sigma shift

    # Identical: PSI should be negligible
    psi_stable, df_stable = calculate_psi(ref, curr_identical, num_bins=10)
    assert psi_stable < 0.08
    assert classify_psi(psi_stable) == "STABLE"
    assert not df_stable.empty
    assert "bin_range" in df_stable.columns
    assert "ref_pct" in df_stable.columns
    assert "curr_pct" in df_stable.columns

    # Shifted: PSI should exceed severe drift threshold
    psi_drift, df_drift = calculate_psi(ref, curr_shifted, num_bins=10)
    assert psi_drift >= 0.25
    assert classify_psi(psi_drift) == "SEVERE_DRIFT"


def test_kolmogorov_smirnov_and_wasserstein():
    """Validates two-sample KS test and Wasserstein distance for continuous divergence."""
    rng = np.random.default_rng(42)
    ref = rng.exponential(scale=5.0, size=1000)
    curr_same = rng.exponential(scale=5.0, size=1000)
    curr_diff = rng.exponential(scale=10.0, size=1000)

    # Identical distribution: KS should NOT reject H0
    ks_same = kolmogorov_smirnov_test(ref, curr_same, alpha=0.05)
    assert not ks_same["drift_detected"]
    assert ks_same["p_value"] > 0.05

    w_same = wasserstein_distance_test(ref, curr_same)
    assert w_same["relative_wasserstein"] < 0.20

    # Different distribution: KS should reject H0
    ks_diff = kolmogorov_smirnov_test(ref, curr_diff, alpha=0.05)
    assert ks_diff["drift_detected"]
    assert ks_diff["p_value"] < 0.01

    w_diff = wasserstein_distance_test(ref, curr_diff)
    assert w_diff["relative_wasserstein"] > 0.50


def test_categorical_divergence():
    """Validates categorical PSI and Jensen-Shannon divergence bounds."""
    ref_cats = np.array(["A", "A", "A", "B", "B", "C"] * 100)
    curr_same = np.array(["A", "A", "A", "B", "B", "C"] * 100)
    curr_shifted = np.array(["A", "B", "C", "C", "C", "C"] * 100)

    # Identical
    res_same = categorical_divergence_test(ref_cats, curr_same)
    assert np.isclose(res_same["psi"], 0.0, atol=1e-3)
    assert np.isclose(res_same["js_divergence"], 0.0, atol=1e-3)
    assert res_same["psi_status"] == "STABLE"

    # Shifted
    res_shift = categorical_divergence_test(ref_cats, curr_shifted)
    assert res_shift["psi"] > 0.10
    assert 0.0 < res_shift["js_divergence"] <= 1.0


def test_tracking_signal_unbiased_vs_biased():
    """Validates Tracking Signal (TS = CFE / MAD) and bias Z-score under zero-mean vs biased errors."""
    rng = np.random.default_rng(42)
    n = 100
    actuals = 10.0 + rng.normal(0, 1.0, size=n)

    # Unbiased forecast: predictions fluctuate symmetrically around actuals
    unbiased_preds = actuals + rng.normal(0, 1.0, size=n)
    cfe_u, mad_u, ts_u = calculate_tracking_signal(actuals, unbiased_preds)
    assert len(ts_u) == n
    err_u = actuals - unbiased_preds
    z_u = np.mean(err_u) / (np.std(err_u, ddof=1) / np.sqrt(n))
    # Unbiased white noise errors must not exhibit statistically significant bias
    assert abs(z_u) < 2.0

    # Systematically biased forecast: model under-forecasts by 3 units consistently
    biased_preds = actuals - 3.0
    cfe_b, mad_b, ts_b = calculate_tracking_signal(actuals, biased_preds)
    # Under-forecasting means actuals > preds -> errors = +3 -> TS grows toward +n
    assert ts_b[-1] > 4.0
    err_b = actuals - biased_preds
    z_b = np.mean(err_b) / (np.std(err_b, ddof=1) / np.sqrt(n) + 1e-6)
    assert z_b > 5.0


def test_cusum_drift_detector():
    """Validates CUSUM control chart: stationary residuals vs structural mean shift."""
    rng = np.random.default_rng(42)
    # 50 stationary residuals around mean 0
    residuals_stationary = rng.normal(0.0, 1.0, size=50)
    cusum_stat = cusum_drift_detector(residuals_stationary, slack_k=0.5, threshold_h=4.0)
    assert not cusum_stat["alarm"]
    assert cusum_stat["change_point_idx"] is None

    # Residuals with sudden mean jump of +2.5 sigma at step 25
    residuals_shifted = np.concatenate([
        rng.normal(0.0, 1.0, size=25),
        rng.normal(2.5, 1.0, size=25),
    ])
    cusum_shift = cusum_drift_detector(residuals_shifted, slack_k=0.5, threshold_h=4.0)
    assert cusum_shift["alarm"]
    assert cusum_shift["change_point_idx"] is not None
    assert 20 <= cusum_shift["change_point_idx"] <= 35


def test_retraining_decision_matrix():
    """Validates multi-tier retraining justification decision logic."""
    baseline_metrics = {"WAPE": 20.0, "MAE": 2.5, "RMSE": 3.8}
    monitor = ModelPerformanceMonitor(baseline_metrics=baseline_metrics, model_name="LightGBM_Test")

    rng = np.random.default_rng(42)
    n = 30
    acts = 50.0 + rng.normal(0, 2.0, size=n)

    # Scenario 1: Stable performance (Delta WAPE < 5%, no persistent bias, PSI < 0.10)
    preds_stable = acts + rng.normal(0, 8.0, size=n)  # WAPE ~16-18%
    dec_stable = monitor.evaluate_retraining_necessity(
        current_actuals=acts,
        current_predictions=preds_stable,
        max_feature_psi=0.04,
    )
    assert dec_stable.decision == RetrainingStatus.NO_ACTION
    assert not dec_stable.is_retraining_justified
    assert dec_stable.urgency == "LOW"

    # Scenario 2: Mild warning (Delta WAPE ~8%, PSI = 0.15)
    preds_warn = acts + rng.normal(0, 10.8, size=n)  # WAPE ~21.6% (+8%)
    dec_warn = monitor.evaluate_retraining_necessity(
        current_actuals=acts,
        current_predictions=preds_warn,
        max_feature_psi=0.15,
    )
    assert dec_warn.decision == RetrainingStatus.WARNING
    assert not dec_warn.is_retraining_justified
    assert dec_warn.urgency == "MEDIUM"

    # Scenario 3: Statistically justified retraining trigger (Delta WAPE > 15%, persistent bias TS > 3.5, Z > 3.0)
    preds_retrain = acts - 12.0  # persistent under-forecasting
    dec_retrain = monitor.evaluate_retraining_necessity(
        current_actuals=acts,
        current_predictions=preds_retrain,
        max_feature_psi=0.28,
    )
    assert dec_retrain.decision == RetrainingStatus.RETRAIN_RECOMMENDED
    assert dec_retrain.is_retraining_justified
    assert dec_retrain.urgency == "HIGH"
    assert dec_retrain.latest_tracking_signal >= 3.5
    assert dec_retrain.bias_z_score >= 3.0

    # Scenario 4: Critical fallback (WAPE triples > +50% increase)
    preds_critical = acts * 0.1  # catastrophic failure
    dec_crit = monitor.evaluate_retraining_necessity(
        current_actuals=acts,
        current_predictions=preds_critical,
    )
    assert dec_crit.decision == RetrainingStatus.CRITICAL_FALLBACK
    assert dec_crit.is_retraining_justified
    assert dec_crit.urgency == "CRITICAL"


def test_data_drift_detector_end_to_end():
    """Integration test: validates DataDriftDetector across features and target regime shift."""
    dates_ref = pd.date_range("2021-01-01", periods=45, freq="D")
    dates_curr = pd.date_range("2021-02-15", periods=30, freq="D")

    # Reference dataframe with normal demand and steady prices
    ref_df = pd.DataFrame({
        "id": ["item_1"] * 45,
        "date": dates_ref,
        "sales": [5.0 + (i % 7) for i in range(45)],
        "sell_price": [2.50] * 45,
        "discount_depth": [0.0] * 45,
        "snap_active": [1 if d.day <= 10 else 0 for d in dates_ref],
    })

    # Current operational dataframe with price increase and promotional discount
    curr_df = pd.DataFrame({
        "id": ["item_1"] * 30,
        "date": dates_curr,
        "sales": [1.0 if (i % 3 == 0) else 0.0 for i in range(30)],  # shifted to intermittent
        "sell_price": [4.00] * 30,  # severe price shift
        "discount_depth": [0.25] * 30,  # discount shift
        "snap_active": [0] * 30,
    })

    detector = DataDriftDetector(
        reference_df=ref_df,
        numerical_features=["sell_price", "discount_depth"],
        categorical_features=["snap_active"],
    )

    feat_report = detector.evaluate_feature_drift(curr_df)
    assert not feat_report.empty
    assert len(feat_report) == 3
    assert "feature_name" in feat_report.columns
    assert "psi" in feat_report.columns
    assert "psi_status" in feat_report.columns

    # sell_price shifted from 2.50 to 4.00 -> severe drift
    price_row = feat_report[feat_report["feature_name"] == "sell_price"].iloc[0]
    assert price_row["psi_status"] in ["MODERATE_DRIFT", "SEVERE_DRIFT"]

    # Target shift evaluation
    target_report = detector.evaluate_target_drift(curr_df, target_col="sales", group_col="id")
    assert target_report["target_psi"] > 0.0
    assert target_report["current_zero_ratio"] > target_report["reference_zero_ratio"]
    assert "mean_percentage_change" in target_report
    assert target_report["regime_change_count"] >= 1
    assert target_report["regime_changes"][0]["reference_regime"] != target_report["regime_changes"][0]["current_regime"]
