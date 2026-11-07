"""Unit and integration test suite for automated model selection and Optuna tuning."""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.optimization.optuna_tuner import (
    LightGBMHyperparameterTuner,
    OptunaTuningConfig,
)
from src.optimization.model_selector import (
    AutoModelSelector,
    ModelSelectionConfig,
    ModelSelectionResult,
)
from src.forecasting.base import DemandForecasterWrapper


@pytest.fixture
def sample_smooth_series():
    """Generates a synthetic smooth demand series with weekly seasonality."""
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    trend = np.linspace(10, 20, 60)
    season = np.sin(np.arange(60) * 2 * np.pi / 7) * 3
    noise = np.random.RandomState(42).normal(0, 1, 60)
    return pd.Series(trend + season + noise, index=dates, name="smooth_sku")


@pytest.fixture
def sample_intermittent_series():
    """Generates a synthetic intermittent demand series with zero-inflation."""
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    rng = np.random.RandomState(42)
    demand = rng.poisson(lam=1.5, size=60) * rng.binomial(n=1, p=0.35, size=60)
    return pd.Series(demand, index=dates, name="intermittent_sku", dtype=float)


@pytest.fixture
def sample_portfolio_df(sample_smooth_series, sample_intermittent_series):
    """Generates a multi-series portfolio wide DataFrame."""
    dates = sample_smooth_series.index
    rng = np.random.RandomState(99)
    erratic = pd.Series(rng.gamma(shape=2, scale=5, size=60), index=dates, name="erratic_sku")
    return pd.DataFrame({
        "smooth_sku": sample_smooth_series,
        "intermittent_sku": sample_intermittent_series,
        "erratic_sku": erratic,
    })


def test_optuna_tuner_optimization(sample_smooth_series):
    """Verifies Optuna Bayesian optimization runs and returns parameterized estimator."""
    config = OptunaTuningConfig(
        n_trials=5,
        horizon=7,
        n_splits=2,
        metric="wape",
        pruning=True,
        random_seed=42,
    )
    tuner = LightGBMHyperparameterTuner(config=config)
    study = tuner.optimize(sample_smooth_series, study_name="test_optuna")

    assert len(study.trials) == 5
    assert tuner.best_params is not None
    assert "learning_rate" in tuner.best_params
    assert "window_length" in tuner.best_params
    assert tuner.best_value is not None
    assert tuner.best_value < 100.0

    # Build and test best estimator
    best_estimator = tuner.get_best_estimator()
    assert isinstance(best_estimator, DemandForecasterWrapper)
    best_estimator.fit(sample_smooth_series.iloc[:50])
    preds = best_estimator.predict(fh=7)
    assert len(preds) == 7
    assert not np.isnan(preds.values).any()


def test_optuna_trials_dataframe(sample_smooth_series):
    """Verifies that trials DataFrame is correctly structured and sorted."""
    config = OptunaTuningConfig(n_trials=4, horizon=7, n_splits=1, random_seed=42)
    tuner = LightGBMHyperparameterTuner(config=config)
    tuner.optimize(sample_smooth_series)
    df_trials = tuner.get_trials_dataframe()

    assert isinstance(df_trials, pd.DataFrame)
    assert len(df_trials) == 4
    assert "value" in df_trials.columns
    # Check sorted ascending by value
    values = df_trials["value"].dropna().tolist()
    assert values == sorted(values)


def test_auto_model_selector_evaluate_series(sample_intermittent_series, sample_smooth_series):
    """Verifies single-series candidate evaluation and champion selection."""
    config = ModelSelectionConfig(horizon=7, n_splits=2, metric="wape", use_regime_routing=True)
    selector = AutoModelSelector(config=config)

    # 1. Intermittent SKU
    res_inter = selector.evaluate_series(sample_intermittent_series, series_id="inter_1")
    assert isinstance(res_inter, ModelSelectionResult)
    assert res_inter.series_id == "inter_1"
    assert res_inter.regime in ("Intermittent", "Lumpy")
    assert res_inter.champion_model in ("Croston", "Seasonal_Naive", "Naive_Last", "Theta")
    assert res_inter.champion_score >= 0.0

    # 2. Smooth SKU
    res_smooth = selector.evaluate_series(sample_smooth_series, series_id="smooth_1")
    assert res_smooth.series_id == "smooth_1"
    assert res_smooth.regime in ("Smooth", "Erratic")
    assert res_smooth.champion_model in ("LightGBM_Reduction", "Theta", "Exponential_Smoothing", "Seasonal_Naive")


def test_auto_model_selector_portfolio(sample_portfolio_df, tmp_path):
    """Verifies portfolio-level model selection, leaderboard, and JSON routing export."""
    config = ModelSelectionConfig(horizon=7, n_splits=2, metric="wape", use_regime_routing=True)
    selector = AutoModelSelector(config=config)

    routing_map, leaderboard_df = selector.select_models_for_portfolio(sample_portfolio_df)

    assert len(routing_map) == 3
    assert set(routing_map.keys()) == {"smooth_sku", "intermittent_sku", "erratic_sku"}
    assert isinstance(leaderboard_df, pd.DataFrame)
    assert len(leaderboard_df) == 3
    expected_cols = [
        "Series ID",
        "Regime",
        "ADI",
        "CV2",
        "Champion Model",
        "Champion WAPE (%)",
        "Runner Up Model",
        "Runner Up WAPE (%)",
        "Selection Margin",
        "Improvement vs Naive (%)",
    ]
    for col in expected_cols:
        assert col in leaderboard_df.columns

    # Test serialization
    json_path = tmp_path / "test_routing.json"
    selector.save_champion_routing(json_path)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "routing_map" in data
    assert "results" in data
    assert data["routing_map"] == routing_map


def test_model_selector_without_regime_routing(sample_smooth_series):
    """Verifies that disabling regime routing benchmarks the full candidate suite."""
    config = ModelSelectionConfig(horizon=7, n_splits=1, metric="wape", use_regime_routing=False)
    selector = AutoModelSelector(config=config)
    res = selector.evaluate_series(sample_smooth_series, series_id="s1")

    # All registered default candidates must have been evaluated
    assert "Croston" in res.all_scores
    assert "Seasonal_Naive" in res.all_scores
    assert "Theta" in res.all_scores
    assert "Exponential_Smoothing" in res.all_scores
    assert "LightGBM_Reduction" in res.all_scores

