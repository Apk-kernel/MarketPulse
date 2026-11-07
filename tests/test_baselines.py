"""Unit and integration tests for Phase 3: sktime baseline forecasters and benchmarking engine."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    calculate_forecast_metrics,
    mean_absolute_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
    mean_absolute_scaled_error,
)
from src.forecasting.base import DemandForecasterWrapper
from src.forecasting.baselines import (
    build_all_baseline_forecasters,
    create_naive_last,
    create_naive_mean,
    create_seasonal_naive,
    create_exponential_smoothing,
    create_theta_forecaster,
    create_auto_arima,
)
from src.evaluation.benchmarking import BaselineBenchmarkEngine


@pytest.fixture
def synthetic_daily_series():
    """Generates a synthetic 70-day series with daily trend and weekly seasonality."""
    dates = pd.date_range("2021-01-01", periods=70, freq="D")
    rng = np.random.default_rng(42)
    # Trend + weekly seasonality (peaking on weekends) + noise
    t = np.arange(70)
    trend = 0.1 * t
    seasonality = np.where(dates.dayofweek.isin([5, 6]), 8.0, 3.0)
    noise = rng.normal(0, 0.5, size=70)
    sales = np.maximum(0.0, 10.0 + trend + seasonality + noise)
    return pd.Series(sales, index=dates, name="sales")


def test_forecast_metrics_correctness():
    """Validates mathematical correctness of forecast metrics and zero-division handling."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])
    y_train = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

    # MAE = mean(|2, -2, 3, -2|) = 9 / 4 = 2.25
    assert np.isclose(mean_absolute_error(y_true, y_pred), 2.25)

    # RMSE = sqrt(mean(4 + 4 + 9 + 4) / 4) = sqrt(21 / 4) = sqrt(5.25) ~ 2.2913
    assert np.isclose(root_mean_squared_error(y_true, y_pred), np.sqrt(5.25))

    # WAPE = sum(|errors|) / sum(actuals) * 100 = 9 / 100 * 100 = 9.0%
    assert np.isclose(weighted_absolute_percentage_error(y_true, y_pred), 9.0)

    # Zero actuals edge case handling
    zero_actuals = np.zeros(5)
    zero_preds = np.zeros(5)
    assert weighted_absolute_percentage_error(zero_actuals, zero_preds) == 0.0

    # Metrics dictionary
    metrics = calculate_forecast_metrics(y_true, y_pred, y_train, sp=1)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "sMAPE" in metrics
    assert "WAPE" in metrics
    assert "MASE" in metrics


def test_demand_forecaster_wrapper_non_negative_clipping(synthetic_daily_series):
    """Tests that DemandForecasterWrapper clips negative forecasts to zero."""
    forecaster = create_naive_last()
    y_train = synthetic_daily_series.iloc[:-7]
    forecaster.fit(y_train)

    preds = forecaster.predict(fh=7)
    assert len(preds) == 7
    assert (preds >= 0.0).all()
    assert forecaster.fit_time_sec >= 0.0
    assert forecaster.predict_time_sec >= 0.0


def test_individual_baseline_forecasters(synthetic_daily_series):
    """Tests fitting and predicting across all required baseline models."""
    y_train = synthetic_daily_series.iloc[:-14]
    y_test = synthetic_daily_series.iloc[-14:]

    models = [
        create_naive_last(),
        create_naive_mean(),
        create_seasonal_naive(sp=7),
        create_exponential_smoothing(sp=7),
        create_theta_forecaster(sp=7),
        create_auto_arima(sp=1),
    ]

    for model in models:
        res = model.fit_and_evaluate(y_train=y_train, y_test=y_test, series_id="test_series", sp=7)
        assert res.model_name == model.name
        assert res.horizon == 14
        assert len(res.predictions) == 14
        assert all(np.isfinite(res.predictions))
        assert all(p >= 0.0 for p in res.predictions)
        assert res.metrics["WAPE"] >= 0.0
        assert res.metrics["RMSE"] >= 0.0


def test_auto_arima_seasonal_default():
    """Validates that create_auto_arima defaults to sp=7 with seasonal configuration enabled."""
    auto_arima = create_auto_arima()
    assert auto_arima.params["sp"] == 7
    # Verify underlying estimator parameters
    if hasattr(auto_arima.estimator, "sp"):
        assert auto_arima.estimator.sp == 7
    if hasattr(auto_arima.estimator, "seasonal"):
        assert auto_arima.estimator.seasonal is True


def test_baseline_benchmark_engine_multi_horizon(synthetic_daily_series):
    """Integration test: validates multi-horizon benchmarking and leaderboard generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_out = tmp_path / "benchmarks.csv"
        md_out = tmp_path / "leaderboard.md"

        forecasters = {
            "naive_last": create_naive_last(),
            "seasonal_naive": create_seasonal_naive(sp=7),
        }

        engine = BaselineBenchmarkEngine(
            forecasters=forecasters,
            horizons=[7, 14],
            sp=7,
        )

        results_df = engine.evaluate_multiple_series({"series_1": synthetic_daily_series})
        assert len(results_df) == 4  # 2 models * 2 horizons

        leaderboard = engine.get_leaderboard(sort_by="WAPE")
        assert not leaderboard.empty
        assert "WAPE" in leaderboard.columns
        assert "horizon" in leaderboard.columns

        engine.save_results(csv_out, md_out)
        assert csv_out.is_file()
        assert md_out.is_file()

        # Reload and check
        reloaded = pd.read_csv(csv_out)
        assert len(reloaded) == 4

