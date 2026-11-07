"""Reusable benchmarking engine for multi-model, multi-horizon time-series evaluation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_forecast_metrics
from src.forecasting.base import DemandForecasterWrapper, ForecastResult
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Evaluation.Benchmarking")


class BaselineBenchmarkEngine:
    """Executes systematic benchmarking of forecasting models across multiple horizons."""

    def __init__(
        self,
        forecasters: Dict[str, DemandForecasterWrapper],
        horizons: Optional[List[int]] = None,
        sp: int = 7,
        fixed_origin: bool = True,
    ):
        """Initializes benchmark engine.

        Args:
            forecasters: Dictionary mapping model_key to DemandForecasterWrapper.
            horizons: List of forecast horizons to evaluate (default: [7, 14, 28]).
            sp: Seasonal periodicity for metrics scaling (default: 7).
            fixed_origin: If True, evaluates all horizons from the exact same forecast origin
                          (T = len(y) - max(horizons)), ensuring fair horizon degradation analysis.
                          If False, shifts cutoff for each horizon (y_train = y[:-h]).
        """
        self.forecasters = forecasters
        self.horizons = sorted(horizons or [7, 14, 28])
        self.sp = sp
        self.fixed_origin = fixed_origin
        self.results: List[ForecastResult] = []

    def evaluate_series(
        self,
        y: pd.Series,
        series_id: str = "series",
    ) -> List[ForecastResult]:
        """Runs all forecasters across configured forecasting horizons for a single series.

        Methodological Guarantees:
        - Uses cloned, un-fitted estimator instances for every model to prevent state leakage.
        - Under fixed_origin=True, fits on identical training history y[:T] and evaluates steps 1..h.
        - Ensures clean temporal separation between train and test intervals.

        Args:
            y: Univariate demand series with DatetimeIndex or RangeIndex.
            series_id: Identifier for the series.

        Returns:
            List of ForecastResult records.
        """
        series_results = []
        n = len(y)
        max_h = max(self.horizons)

        if n <= max_h + self.sp:
            logger.warning(
                f"Series '{series_id}' has length {n}, insufficient for max horizon {max_h} + sp {self.sp}. Skipping."
            )
            return series_results

        if self.fixed_origin:
            # Fixed forecast origin: train on y[:-max_h], test on y[-max_h:]
            y_train = y.iloc[:-max_h]
            y_test_full = y.iloc[-max_h:]

            for model_key, model_wrapper in self.forecasters.items():
                logger.info(
                    f"Evaluating [{model_wrapper.name}] on '{series_id}' (fixed origin T={len(y_train)}, max_h={max_h})..."
                )
                try:
                    # Clone estimator to guarantee zero state or cache persistence
                    cloned_model = model_wrapper.clone()
                    cloned_model.fit(y=y_train, fh=max_h)
                    # Predict full maximum horizon
                    full_predictions = cloned_model.predict(fh=max_h)
                    full_preds_arr = np.asarray(full_predictions, dtype=float)
                    full_acts_arr = np.asarray(y_test_full, dtype=float)

                    for h in self.horizons:
                        sub_preds = full_preds_arr[:h]
                        sub_acts = full_acts_arr[:h]
                        sub_test = y_test_full.iloc[:h]

                        metrics = calculate_forecast_metrics(
                            y_true=sub_acts,
                            y_pred=sub_preds,
                            y_train=np.asarray(y_train, dtype=float),
                            sp=self.sp,
                        )

                        timestamps = [str(t) for t in sub_test.index] if hasattr(sub_test, "index") else None
                        res = ForecastResult(
                            model_name=cloned_model.name,
                            series_id=series_id,
                            horizon=h,
                            predictions=[round(float(p), 4) for p in sub_preds],
                            actuals=[round(float(a), 4) for a in sub_acts],
                            timestamps=timestamps,
                            metrics=metrics,
                            fit_time_sec=cloned_model.fit_time_sec,
                            predict_time_sec=cloned_model.predict_time_sec,
                            model_config=cloned_model.params,
                        )
                        series_results.append(res)
                        self.results.append(res)
                except Exception as e:
                    logger.error(f"Error evaluating model '{model_wrapper.name}' on '{series_id}': {e}")

        else:
            # Shifted cutoff per horizon: y_train = y[:-h], y_test = y[-h:]
            for h in self.horizons:
                y_train = y.iloc[:-h]
                y_test = y.iloc[-h:]

                for model_key, model_wrapper in self.forecasters.items():
                    logger.info(
                        f"Evaluating [{model_wrapper.name}] on series '{series_id}' (horizon={h} days)..."
                    )
                    try:
                        cloned_model = model_wrapper.clone()
                        res = cloned_model.fit_and_evaluate(
                            y_train=y_train,
                            y_test=y_test,
                            series_id=series_id,
                            sp=self.sp,
                        )
                        series_results.append(res)
                        self.results.append(res)
                    except Exception as e:
                        logger.error(
                            f"Error evaluating model '{model_wrapper.name}' on '{series_id}' at horizon {h}: {e}"
                        )

        return series_results

    def evaluate_multiple_series(
        self,
        series_dict: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Evaluates all forecasters across multiple time series."""
        logger.info(
            f"Starting benchmark evaluation across {len(series_dict)} series "
            f"and horizons {self.horizons} (fixed_origin={self.fixed_origin})..."
        )
        for s_id, y_series in series_dict.items():
            self.evaluate_series(y=y_series, series_id=s_id)

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Converts accumulated ForecastResults into a structured pandas DataFrame."""
        rows = []
        for r in self.results:
            row = {
                "model_name": r.model_name,
                "series_id": r.series_id,
                "horizon": r.horizon,
                "fit_time_sec": r.fit_time_sec,
                "predict_time_sec": r.predict_time_sec,
            }
            for metric_name, metric_val in r.metrics.items():
                row[metric_name] = metric_val
            rows.append(row)

        return pd.DataFrame(rows)

    def get_leaderboard(
        self,
        sort_by: str = "WAPE",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Aggregates metrics across series and horizons to generate a model leaderboard."""
        df = self.get_results_dataframe()
        if df.empty:
            return pd.DataFrame()

        metrics_cols = [c for c in ["MAE", "RMSE", "sMAPE", "WAPE", "MASE", "fit_time_sec"] if c in df.columns]
        leaderboard = (
            df.groupby(["model_name", "horizon"])[metrics_cols]
            .mean()
            .reset_index()
        )
        if sort_by in leaderboard.columns:
            leaderboard = leaderboard.sort_values(by=["horizon", sort_by], ascending=[True, ascending])

        return leaderboard

    def save_results(
        self,
        output_csv_path: Union[str, Path],
        output_leaderboard_md: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Saves benchmark results to CSV and markdown leaderboard."""
        path = Path(output_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results_df = self.get_results_dataframe()
        results_df.to_csv(path, index=False)
        logger.info(f"Saved benchmark results to {path}")

        if output_leaderboard_md:
            md_path = Path(output_leaderboard_md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            leaderboard = self.get_leaderboard()
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# MarketPulse: Baseline Forecasting Benchmark Leaderboard\n\n")
                f.write(leaderboard.to_markdown(index=False))
            logger.info(f"Saved leaderboard markdown to {md_path}")

        return path
