"""Comprehensive unit and integration test suite for Phase 7: Probabilistic Forecasting."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from src.evaluation.metrics import (
    continuous_ranked_probability_score,
    mean_pinball_loss,
    mean_prediction_interval_width,
    pinball_loss,
    prediction_interval_coverage_probability,
    winkler_score,
)
from src.forecasting.baselines import create_seasonal_naive
from src.forecasting.probabilistic import (
    DEFAULT_QUANTILES,
    ConformalIntervalForecaster,
    IntermittentProbabilisticForecaster,
    QuantileLightGBMForecaster,
    enforce_monotonic_quantiles,
)


@pytest.fixture
def synthetic_time_series():
    """Generates synthetic 75-day time series with trend and weekly seasonality."""
    dates = pd.date_range("2021-01-01", periods=75, freq="D")
    rng = np.random.default_rng(42)
    t = np.arange(75)
    trend = 0.05 * t
    seasonality = np.where(dates.dayofweek.isin([5, 6]), 6.0, 2.0)
    noise = rng.normal(0, 1.0, size=75)
    sales = np.maximum(0.0, 10.0 + trend + seasonality + noise)
    return pd.Series(sales, index=dates, name="sales")


@pytest.fixture
def intermittent_series():
    """Generates intermittent demand series with 65% zero sales periods."""
    dates = pd.date_range("2021-01-01", periods=90, freq="D")
    rng = np.random.default_rng(123)
    is_positive = rng.uniform(0, 1, size=90) > 0.65
    sizes = rng.poisson(lam=4.0, size=90) + 1.0
    sales = np.where(is_positive, sizes, 0.0)
    return pd.Series(sales, index=dates, name="sales")


# ============================================================================
# 1. Probabilistic Metric Tests
# ============================================================================

def test_pinball_loss_exactness():
    """Validates pinball loss against known hand-calculated values."""
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([8.0, 25.0])

    # For y >= q (10 >= 8), diff = 2.0, loss = alpha * 2.0
    # For y < q (20 < 25), diff = -5.0, loss = (1 - alpha) * 5.0
    # At alpha = 0.9:
    # 0.9 * 2.0 = 1.8
    # 0.1 * 5.0 = 0.5
    # Mean = (1.8 + 0.5) / 2 = 1.15
    loss_09 = pinball_loss(y_true, y_pred, alpha=0.9)
    assert np.isclose(loss_09, 1.15, atol=1e-5)

    # At alpha = 0.5 (median / half MAE):
    # 0.5 * 2.0 + 0.5 * 5.0 = 3.5 -> Mean = 1.75
    loss_05 = pinball_loss(y_true, y_pred, alpha=0.5)
    assert np.isclose(loss_05, 1.75, atol=1e-5)


def test_picp_and_mpiw_bounds():
    """Verifies coverage and width calculations."""
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    lower = np.array([8.0, 12.0, 22.0, 20.0])
    upper = np.array([12.0, 18.0, 28.0, 30.0])

    # 10 in [8, 12] -> Yes
    # 15 in [12, 18] -> Yes
    # 20 in [22, 28] -> No (below lower)
    # 25 in [20, 30] -> Yes
    # PICP = 3 / 4 * 100 = 75.0%
    picp = prediction_interval_coverage_probability(y_true, lower, upper)
    assert np.isclose(picp, 75.0, atol=1e-5)

    # Widths: [4, 6, 6, 10] -> Mean = 26 / 4 = 6.5
    mpiw = mean_prediction_interval_width(lower, upper)
    assert np.isclose(mpiw, 6.5, atol=1e-5)


def test_winkler_score_and_crps():
    """Verifies Winkler score penalty behavior and CRPS non-negativity."""
    y_true = np.array([10.0, 25.0])
    lower = np.array([8.0, 10.0])
    upper = np.array([12.0, 20.0])

    score = winkler_score(y_true, lower, upper, nominal_coverage=0.90)
    assert score > 0.0

    quantiles_df = pd.DataFrame({
        0.10: [7.0, 9.0],
        0.50: [10.0, 21.0],
        0.90: [13.0, 26.0],
    })
    crps = continuous_ranked_probability_score(y_true, quantiles_df)
    assert crps > 0.0


# ============================================================================
# 2. Monotonic Rearrangement (Non-Crossing Quantiles)
# ============================================================================

def test_enforce_monotonic_quantiles():
    """Verifies sorting across quantiles to eliminate quantile crossing."""
    crossed_df = pd.DataFrame({
        0.10: [15.0, 12.0],  # higher than 0.50!
        0.50: [10.0, 14.0],
        0.90: [8.0, 20.0],   # lower than 0.10 and 0.50!
    })

    clean_df = enforce_monotonic_quantiles(crossed_df, non_negative=True)
    vals = clean_df.values

    # Strict monotonicity along columns
    assert (np.diff(vals, axis=1) >= -1e-8).all(), "Quantile crossing still present after rearrangement!"
    assert (vals >= 0.0).all(), "Negative value detected in non-negative quantiles!"


# ============================================================================
# 3. Quantile LightGBM Forecaster
# ============================================================================

def test_quantile_lightgbm_fit_predict(synthetic_time_series):
    """Verifies multi-quantile LightGBM fitting, monotonic quantiles, and interval extraction."""
    forecaster = QuantileLightGBMForecaster(
        quantiles=[0.10, 0.50, 0.90],
        lags=[1, 7],
        rolling_windows=[7],
        n_estimators=30,
        random_seed=42,
    )
    forecaster.fit(synthetic_time_series)

    horizon = 14
    q_df = forecaster.predict_quantiles(horizon=horizon)

    assert q_df.shape == (horizon, 3)
    assert list(q_df.columns) == [0.10, 0.50, 0.90]

    # Zero quantile crossing
    diffs = np.diff(q_df.values, axis=1)
    assert (diffs >= -1e-6).all(), "Detected quantile crossing in QuantileLightGBM!"
    assert (q_df.values >= 0.0).all(), "Detected negative demand in QuantileLightGBM!"

    # Interval extraction
    lower, upper = forecaster.predict_intervals(coverage=0.80, horizon=horizon)
    assert len(lower) == horizon
    assert len(upper) == horizon
    assert (upper.values >= lower.values).all()


# ============================================================================
# 4. Intermittent Probabilistic Forecaster
# ============================================================================

def test_intermittent_probabilistic_forecaster(intermittent_series):
    """Verifies zero-inflated quantiles on intermittent series."""
    forecaster = IntermittentProbabilisticForecaster()
    forecaster.fit(intermittent_series)

    # In our fixture, zeros are ~65%, so occurrence probability is ~35%
    assert 0.20 <= forecaster.occurrence_prob_ <= 0.50

    q_df = forecaster.predict_quantiles(horizon=10, quantiles=[0.10, 0.50, 0.90])
    assert q_df.shape == (10, 3)

    # With 65% zeros, quantile 0.10 and 0.50 must be exactly 0.0!
    assert (q_df[0.10].values == 0.0).all(), "Low quantile of intermittent series must be zero!"
    assert (q_df[0.50].values == 0.0).all(), "Median of high-intermittency series must be zero!"

    # Upper quantile 0.90 captures positive order size
    assert (q_df[0.90].values > 0.0).all()

    # Monotonicity
    assert (np.diff(q_df.values, axis=1) >= 0.0).all()


# ============================================================================
# 5. Conformal Interval Forecaster
# ============================================================================

def test_conformal_interval_forecaster(synthetic_time_series):
    """Verifies distribution-free conformal prediction intervals."""
    base_wrapper = create_seasonal_naive(sp=7)
    conformal = ConformalIntervalForecaster(
        forecaster=base_wrapper.estimator,
        nominal_coverage=0.90,
    )
    conformal.fit(synthetic_time_series, val_fraction=0.25)

    assert conformal.calibration_cutoff_ > 0.0

    point, lower, upper = conformal.predict_intervals(horizon=14)
    assert len(point) == 14
    assert len(lower) == 14
    assert len(upper) == 14

    assert (upper.values >= point.values).all()
    assert (point.values >= lower.values).all()
    assert (lower.values >= 0.0).all()  # Non-negative demand constraint

