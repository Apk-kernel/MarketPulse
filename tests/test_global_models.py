"""Unit and integration tests for Phase 5: Global/Local ML forecasting and reduction models."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.lag_features import create_lag_features
from src.features.rolling_features import (
    create_rolling_features,
    create_price_dynamics_features,
)
from src.forecasting.global_models import (
    get_base_regressor,
    create_reduction_forecaster,
    build_ml_reduction_models,
    prepare_global_panel_dataframe,
)


@pytest.fixture
def sample_sales_series():
    """Generates a synthetic 60-day time series."""
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    sales = [float(i % 10 + 2) for i in range(60)]
    return pd.Series(sales, index=dates, name="sales")


def test_lag_features_correctness_and_zero_leakage():
    """Validates lag feature values and asserts zero future lookahead."""
    df = pd.DataFrame({
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    df_lags = create_lag_features(df, target_col="sales", lags=[1, 2])

    assert "lag_1" in df_lags.columns
    assert "lag_2" in df_lags.columns

    # Row 0: lags must be NaN
    assert np.isnan(df_lags.loc[0, "lag_1"])
    # Row 1: lag_1 is row 0 (10.0)
    assert df_lags.loc[1, "lag_1"] == 10.0
    # Row 2: lag_1 is row 1 (20.0), lag_2 is row 0 (10.0)
    assert df_lags.loc[2, "lag_1"] == 20.0
    assert df_lags.loc[2, "lag_2"] == 10.0

    # Negative or zero lag must raise ValueError
    with pytest.raises(ValueError, match="Lag must be >= 1"):
        create_lag_features(df, lags=[0])


def test_rolling_features_leakage_prevention():
    """Validates that rolling features strictly use shift >= 1 to prevent current-step leakage."""
    # Data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    df = pd.DataFrame({"sales": list(range(1, 11))})
    df_roll = create_rolling_features(df, target_col="sales", mean_windows=[3], std_windows=[3], shift=1)

    assert "rolling_mean_3" in df_roll.columns
    assert "rolling_std_3" in df_roll.columns

    # For index 3 (value=4), the window should be the 3 values BEFORE index 3: [1, 2, 3] -> mean = 2.0
    assert np.isclose(df_roll.loc[3, "rolling_mean_3"], 2.0)
    # The actual value at index 3 (4) must NOT be included in its rolling mean
    assert df_roll.loc[3, "rolling_mean_3"] != np.mean([2, 3, 4])

    with pytest.raises(ValueError, match="Shift must be >= 1"):
        create_rolling_features(df, shift=0)


def test_price_dynamics_features():
    """Validates previous price, price difference, and rolling price features."""
    df = pd.DataFrame({
        "sell_price": [10.0, 10.0, 8.0, 8.0, 12.0],
    })
    df_dyn = create_price_dynamics_features(df, price_col="sell_price")

    assert "prev_price" in df_dyn.columns
    assert "price_diff" in df_dyn.columns
    assert "rolling_price_mean_7" in df_dyn.columns

    # Row 2: price dropped from 10.0 to 8.0 -> diff = -2.0
    assert np.isclose(df_dyn.loc[2, "price_diff"], -2.0)


def test_candidate_regressors_instantiation():
    """Tests instantiation of all candidate ML regressors."""
    lgb = get_base_regressor("lightgbm", random_state=42)
    hgb = get_base_regressor("hist_gradient_boosting", random_state=42)
    rf = get_base_regressor("random_forest", random_state=42)

    assert lgb.__class__.__name__ == "LGBMRegressor"
    assert hgb.__class__.__name__ == "HistGradientBoostingRegressor"
    assert rf.__class__.__name__ == "RandomForestRegressor"


def test_reduction_forecasters_fit_predict(sample_sales_series):
    """Tests fitting and multi-step prediction for recursive and direct ML reduction forecasters."""
    y_train = sample_sales_series.iloc[:-7]
    y_test = sample_sales_series.iloc[-7:]

    # 1. LightGBM Recursive
    lgb_rec = create_reduction_forecaster("lightgbm", strategy="recursive", window_length=7)
    res_rec = lgb_rec.fit_and_evaluate(y_train=y_train, y_test=y_test, series_id="test_lgb_rec")
    assert res_rec.horizon == 7
    assert len(res_rec.predictions) == 7
    assert all(np.isfinite(res_rec.predictions))
    assert all(p >= 0.0 for p in res_rec.predictions)

    # 2. HistGradientBoosting Direct
    hgb_dir = create_reduction_forecaster("hist_gradient_boosting", strategy="direct", window_length=7)
    res_dir = hgb_dir.fit_and_evaluate(y_train=y_train, y_test=y_test, series_id="test_hgb_dir")
    assert res_dir.horizon == 7
    assert len(res_dir.predictions) == 7
    assert all(np.isfinite(res_dir.predictions))


def test_prepare_global_panel_dataframe(sample_sales_series):
    """Tests multi-index panel formatting required by sktime for pooling='global'."""
    series_dict = {
        "series_1": sample_sales_series,
        "series_2": sample_sales_series * 1.5,
    }
    panel_df = prepare_global_panel_dataframe(series_dict, target_col="sales")
    assert isinstance(panel_df.index, pd.MultiIndex)
    assert panel_df.index.names[0] == "instance"
    assert "series_1" in panel_df.index.levels[0]
    assert "series_2" in panel_df.index.levels[0]
    assert len(panel_df) == len(sample_sales_series) * 2

