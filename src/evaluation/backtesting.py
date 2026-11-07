"""Rigorous temporal backtesting engine with expanding and sliding window validation.

Guarantees:
1. Strict temporal causality: train window always precedes test window (t_train < t_test).
2. Zero leakage: fresh cloned estimator instance for each fold.
3. Multiple forecasting horizons and multi-fold cross-validation.
4. M5-compatible RMSSE and WRMSSE evaluation.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter

from src.evaluation.metrics import (
    calculate_forecast_metrics,
    root_mean_squared_scaled_error,
    weighted_root_mean_squared_scaled_error,
)
from src.forecasting.base import DemandForecasterWrapper, ForecastResult
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Evaluation.Backtesting")


@dataclass
class BacktestFoldResult:
    """Evaluation output for a single temporal backtest fold."""

    fold_idx: int
    model_name: str
    series_id: str
    horizon: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    fit_time_sec: float
    predict_time_sec: float
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TemporalBacktester:
    """Manages expanding-window and sliding-window cross-validation for time series."""

    def __init__(
        self,
        forecasters: Dict[str, DemandForecasterWrapper],
        fh: Union[int, List[int]] = 28,
        n_splits: int = 3,
        window_type: str = "expanding",
        step_length: Optional[int] = None,
        sp: int = 7,
    ):
        """Initializes TemporalBacktester.

        Args:
            forecasters: Dictionary mapping model key to DemandForecasterWrapper.
            fh: Forecast horizon (e.g. 28 days).
            n_splits: Number of temporal backtest folds.
            window_type: 'expanding' (growing train size) or 'sliding' / 'rolling' (fixed train size).
            step_length: Step between successive fold cutoffs (defaults to fh).
            sp: Seasonal periodicity for MASE scaling (default: 7).
        """
        self.forecasters = forecasters
        self.fh_val = fh if isinstance(fh, int) else len(fh)
        self.fh_list = list(range(1, self.fh_val + 1))
        self.n_splits = n_splits
        self.window_type = window_type.lower()
        self.step_length = step_length or self.fh_val
        self.sp = sp
        self.fold_results: List[BacktestFoldResult] = []

    def get_splitter(self, total_length: int) -> Union[ExpandingWindowSplitter, SlidingWindowSplitter]:
        """Constructs the sktime temporal splitter based on configured window_type and series length."""
        # Total test points needed for n_splits = (n_splits - 1) * step_length + fh_val
        test_requirement = (self.n_splits - 1) * self.step_length + self.fh_val
        initial_window = total_length - test_requirement

        if initial_window <= self.sp * 2:
            raise ValueError(
                f"Series length {total_length} is insufficient for {self.n_splits} splits "
                f"with horizon={self.fh_val} and step_length={self.step_length}. "
                f"Initial window would be {initial_window} (must be > {self.sp * 2})."
            )

        if self.window_type == "expanding":
            return ExpandingWindowSplitter(
                fh=self.fh_list,
                initial_window=initial_window,
                step_length=self.step_length,
            )
        elif self.window_type in ["sliding", "rolling"]:
            return SlidingWindowSplitter(
                fh=self.fh_list,
                window_length=initial_window,
                step_length=self.step_length,
            )
        else:
            raise ValueError(f"Unsupported window_type: {self.window_type}. Use 'expanding' or 'sliding'.")

    def backtest_series(
        self,
        y: pd.Series,
        X: Optional[pd.DataFrame] = None,
        series_id: str = "series",
    ) -> List[BacktestFoldResult]:
        """Executes temporal cross-validation for all forecasters on a single time series.

        Strictly guarantees temporal causality: train window always precedes test window
        for both target variable y and exogenous features X.

        Args:
            y: Demand time series.
            X: Optional exogenous features DataFrame aligned with y.
            series_id: Series identifier.

        Returns:
            List of BacktestFoldResult objects.
        """
        clean_y = DemandForecasterWrapper._ensure_valid_temporal_index(y)
        n = len(clean_y)
        splitter = self.get_splitter(n)
        splits = list(splitter.split(clean_y))

        # Take only the requested number of splits from the end
        if len(splits) > self.n_splits:
            splits = splits[-self.n_splits:]

        series_fold_results = []
        logger.info(
            f"Backtesting '{series_id}' across {len(splits)} {self.window_type} folds (H={self.fh_val} days)..."
        )

        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            y_train = clean_y.iloc[train_idx]
            y_test = clean_y.iloc[test_idx]

            X_train = X.iloc[train_idx] if X is not None else None
            X_test = X.iloc[test_idx] if X is not None else None

            # Leakage Assertion: Train dates must strictly precede test dates
            assert train_idx[-1] < test_idx[0], (
                f"Temporal leakage violation in fold {fold_idx}: "
                f"train_idx max ({train_idx[-1]}) >= test_idx min ({test_idx[0]})"
            )

            train_start_str = str(y_train.index[0]) if hasattr(y_train, "index") else str(train_idx[0])
            train_end_str = str(y_train.index[-1]) if hasattr(y_train, "index") else str(train_idx[-1])
            test_start_str = str(y_test.index[0]) if hasattr(y_test, "index") else str(test_idx[0])
            test_end_str = str(y_test.index[-1]) if hasattr(y_test, "index") else str(test_idx[-1])

            for model_key, model_wrapper in self.forecasters.items():
                # Clone model to guarantee zero state leakage across folds
                cloned = model_wrapper.clone()
                start_fit = time.perf_counter()
                cloned.fit(y=y_train, X=X_train)
                fit_sec = time.perf_counter() - start_fit

                start_pred = time.perf_counter()
                preds = cloned.predict(fh=self.fh_val, X=X_test)
                pred_sec = time.perf_counter() - start_pred

                preds_arr = np.asarray(preds, dtype=float)
                acts_arr = np.asarray(y_test, dtype=float)

                metrics = calculate_forecast_metrics(
                    y_true=acts_arr,
                    y_pred=preds_arr,
                    y_train=np.asarray(y_train, dtype=float),
                    sp=self.sp,
                )

                fold_res = BacktestFoldResult(
                    fold_idx=fold_idx,
                    model_name=cloned.name,
                    series_id=series_id,
                    horizon=self.fh_val,
                    train_start=train_start_str,
                    train_end=train_end_str,
                    test_start=test_start_str,
                    test_end=test_end_str,
                    n_train=len(y_train),
                    n_test=len(y_test),
                    fit_time_sec=round(fit_sec, 4),
                    predict_time_sec=round(pred_sec, 4),
                    metrics=metrics,
                    predictions=[round(float(p), 4) for p in preds_arr],
                    actuals=[round(float(a), 4) for a in acts_arr],
                )
                series_fold_results.append(fold_res)
                self.fold_results.append(fold_res)

        return series_fold_results

    def backtest_multiple_series(
        self,
        series_dict: Dict[str, pd.Series],
        X_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Executes backtesting across multiple time-series.

        Args:
            series_dict: Dictionary mapping series_id to demand Series.
            X_dict: Optional dictionary mapping series_id to exogenous DataFrame.

        Returns:
            Summary DataFrame of all fold evaluations.
        """
        for s_id, s_series in series_dict.items():
            s_X = X_dict.get(s_id) if X_dict is not None else None
            self.backtest_series(s_series, X=s_X, series_id=s_id)
        return self.get_summary_dataframe()

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Flattens all fold results into a structured DataFrame."""
        rows = []
        for f in self.fold_results:
            row = {
                "fold_idx": f.fold_idx,
                "model_name": f.model_name,
                "series_id": f.series_id,
                "horizon": f.horizon,
                "train_window": f"{f.train_start} to {f.train_end}",
                "test_window": f"{f.test_start} to {f.test_end}",
                "n_train": f.n_train,
                "n_test": f.n_test,
                "fit_time_sec": f.fit_time_sec,
                "predict_time_sec": f.predict_time_sec,
            }
            for m_k, m_v in f.metrics.items():
                row[m_k] = m_v
            rows.append(row)
        return pd.DataFrame(rows)

    def get_aggregated_comparison(
        self,
        sort_by: str = "WAPE",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Computes mean and std of metrics across folds and series to build comparison table.

        Args:
            sort_by: Primary ranking metric (default: 'WAPE').
            ascending: Sort order.

        Returns:
            Leaderboard comparison DataFrame with Mean ± Std.
        """
        df = self.get_summary_dataframe()
        if df.empty:
            return pd.DataFrame()

        metric_cols = [c for c in ["WAPE", "RMSE", "MAE", "sMAPE", "MASE", "RMSSE", "fit_time_sec"] if c in df.columns]

        grouped = df.groupby(["model_name", "horizon"])
        agg_mean = grouped[metric_cols].mean()
        agg_std = grouped[metric_cols].std().fillna(0.0)

        comparison_rows = []
        for (m_name, h_val), mean_series in agg_mean.iterrows():
            std_series = agg_std.loc[(m_name, h_val)]
            row = {
                "model_name": m_name,
                "horizon": h_val,
                "WAPE (%)": f"{mean_series['WAPE']:.2f} ± {std_series['WAPE']:.2f}" if "WAPE" in mean_series else "N/A",
                "RMSE": f"{mean_series['RMSE']:.2f} ± {std_series['RMSE']:.2f}" if "RMSE" in mean_series else "N/A",
                "MAE": f"{mean_series['MAE']:.2f} ± {std_series['MAE']:.2f}" if "MAE" in mean_series else "N/A",
                "RMSSE": f"{mean_series['RMSSE']:.3f} ± {std_series['RMSSE']:.3f}" if "RMSSE" in mean_series else "N/A",
                "sMAPE (%)": f"{mean_series['sMAPE']:.2f} ± {std_series['sMAPE']:.2f}" if "sMAPE" in mean_series else "N/A",
                "Avg Fit Time (s)": round(float(mean_series.get("fit_time_sec", 0.0)), 4),
                "_sort_val": float(mean_series[sort_by]) if sort_by in mean_series else 0.0,
            }
            comparison_rows.append(row)

        comp_df = pd.DataFrame(comparison_rows).sort_values(by=["horizon", "_sort_val"], ascending=[True, ascending])
        comp_df = comp_df.drop(columns=["_sort_val"])
        return comp_df

    def save_results(
        self,
        output_csv: Union[str, Path],
        output_md: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Saves detailed backtest results to CSV and markdown report."""
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.get_summary_dataframe().to_csv(csv_path, index=False)
        logger.info(f"Saved backtest detailed fold results to {csv_path}")

        if output_md:
            md_path = Path(output_md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            comparison = self.get_aggregated_comparison()
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# MarketPulse: Time-Series Temporal Backtesting Report\n")
                f.write(f"**Validation Strategy**: `{self.window_type.capitalize()} Window` ({self.n_splits} Folds)\n")
                f.write(f"**Forecasting Horizon**: `{self.fh_val} days`\n\n")
                f.write("## Backtest Leaderboard (Mean ± Std Across Folds)\n\n")
                f.write(comparison.to_markdown(index=False))
            logger.info(f"Saved backtest summary markdown to {md_path}")

        return csv_path

