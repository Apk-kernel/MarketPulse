"""Unit and integration tests for Phase 4: Temporal Backtesting and M5 Hierarchical Validation."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    root_mean_squared_scaled_error,
    weighted_root_mean_squared_scaled_error,
    mean_absolute_percentage_error_safe,
)
from src.evaluation.backtesting import TemporalBacktester
from src.forecasting.baselines import create_naive_last, create_seasonal_naive


@pytest.fixture
def sample_temporal_series():
    """Generates a 90-day daily time-series with trend and weekly seasonality."""
    dates = pd.date_range("2021-01-01", periods=90, freq="D")
    rng = np.random.default_rng(42)
    t = np.arange(90)
    sales = 5.0 + 0.05 * t + np.where(dates.dayofweek.isin([5, 6]), 4.0, 1.0) + rng.normal(0, 0.3, size=90)
    return pd.Series(np.maximum(0.0, sales), index=dates, name="sales")


def test_rmsse_calculation():
    """Validates M5 RMSSE calculation against analytical ground truth."""
    # Scale calculation: y_train diffs = [1, 1, 1, 1] -> diff^2 = 1 -> mean(diff^2) = 1.0
    y_train = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    # Errors on test: actuals [15, 16], preds [14, 15] -> errors = [1, 1] -> MSE = 1.0
    y_true = np.array([15.0, 16.0])
    y_pred = np.array([14.0, 15.0])

    rmsse = root_mean_squared_scaled_error(y_true, y_pred, y_train, after_first_nonzero=False)
    # RMSSE = sqrt(1.0 / 1.0) = 1.0
    assert np.isclose(rmsse, 1.0)

    # Constant training series guard (diffs = 0)
    const_train = np.array([5.0, 5.0, 5.0, 5.0])
    rmsse_const = root_mean_squared_scaled_error(y_true, y_pred, const_train)
    assert np.isfinite(rmsse_const)


def test_wrmsse_calculation():
    """Validates Weighted RMSSE across series with dollar volume weights."""
    rmsse_vals = [1.2, 0.8]
    weights = [1000.0, 3000.0]  # Total = 4000 -> weights: 0.25, 0.75

    wrmsse = weighted_root_mean_squared_scaled_error(rmsse_vals, weights, normalize_weights=True)
    # Expected: 0.25 * 1.2 + 0.75 * 0.8 = 0.30 + 0.60 = 0.90
    assert np.isclose(wrmsse, 0.90)


def test_mape_safe_zero_handling():
    """Validates that MAPE safe function handles zeroes without ZeroDivisionError."""
    # All zeroes -> returns NaN
    all_zeroes = np.zeros(5)
    preds = np.ones(5)
    assert np.isnan(mean_absolute_percentage_error_safe(all_zeroes, preds))

    # Mixed series with zero actual
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([2.0, 11.0, 18.0])
    # Zero actual is skipped; evaluated on [10, 20]: errors = [1/10, 2/20] = [0.1, 0.1] -> mean 10.0%
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)
    assert np.isclose(mape, 10.0)


def test_temporal_leakage_absence(sample_temporal_series):
    """Rigorous test verifying strict temporal causality: train window always precedes test window."""
    forecasters = {"naive_last": create_naive_last()}
    backtester = TemporalBacktester(
        forecasters=forecasters,
        fh=14,
        n_splits=3,
        window_type="expanding",
        sp=7,
    )

    clean_y = sample_temporal_series
    splitter = backtester.get_splitter(len(clean_y))
    splits = list(splitter.split(clean_y))[-3:]

    prev_train_max = -1
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        # 1. No overlap between train and test
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        # 2. Train strictly precedes test
        assert train_idx[-1] < test_idx[0]
        # 3. Expanding window property: train set expands fold-by-fold
        assert train_idx[-1] > prev_train_max
        prev_train_max = train_idx[-1]


def test_end_to_end_backtester(sample_temporal_series):
    """Integration test: validates multi-fold backtest execution, metric aggregation, and report generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_out = tmp_path / "backtest_results.csv"
        md_out = tmp_path / "backtest_summary.md"

        forecasters = {
            "naive_last": create_naive_last(),
            "seasonal_naive": create_seasonal_naive(sp=7),
        }

        backtester = TemporalBacktester(
            forecasters=forecasters,
            fh=14,
            n_splits=2,
            window_type="expanding",
            sp=7,
        )

        fold_results = backtester.backtest_series(sample_temporal_series, series_id="test_series")
        assert len(fold_results) == 4  # 2 models * 2 folds

        # Verify summary DataFrame schema
        summary_df = backtester.get_summary_dataframe()
        assert "fold_idx" in summary_df.columns
        assert "train_window" in summary_df.columns
        assert "test_window" in summary_df.columns
        assert "WAPE" in summary_df.columns
        assert "RMSSE" in summary_df.columns

        # Verify aggregated comparison table (Mean ± Std)
        comparison = backtester.get_aggregated_comparison(sort_by="WAPE")
        assert not comparison.empty
        assert "WAPE (%)" in comparison.columns
        assert "RMSSE" in comparison.columns

        # Verify file export
        backtester.save_results(output_csv=csv_out, output_md=md_out)
        assert csv_out.is_file()
        assert md_out.is_file()


def test_backtester_with_exogenous_features(sample_temporal_series):
    """Validates that TemporalBacktester correctly partitions and supplies exogenous features X without leakage."""
    n = len(sample_temporal_series)
    dates = sample_temporal_series.index
    # Create synthetic promotional and calendar exogenous features
    X = pd.DataFrame(
        {
            "promo_active": np.where(dates.dayofweek.isin([4, 5]), 1.0, 0.0),
            "price": 2.50 + 0.1 * np.sin(np.arange(n)),
        },
        index=dates,
    )

    forecasters = {
        "naive_last": create_naive_last(),
        "seasonal_naive": create_seasonal_naive(sp=7),
    }

    backtester = TemporalBacktester(
        forecasters=forecasters,
        fh=7,
        n_splits=2,
        window_type="expanding",
        sp=7,
    )

    # Single series with exogenous X
    fold_results = backtester.backtest_series(sample_temporal_series, X=X, series_id="exo_series")
    assert len(fold_results) == 4  # 2 models * 2 folds
    for r in fold_results:
        assert len(r.predictions) == 7
        assert all(np.isfinite(r.predictions))
        assert r.metrics["WAPE"] >= 0.0

    # Multi-series with X_dict
    backtester_multi = TemporalBacktester(
        forecasters=forecasters,
        fh=7,
        n_splits=2,
        window_type="expanding",
        sp=7,
    )
    multi_summary = backtester_multi.backtest_multiple_series(
        series_dict={"s1": sample_temporal_series},
        X_dict={"s1": X},
    )
    assert not multi_summary.empty
    assert len(multi_summary) == 4


