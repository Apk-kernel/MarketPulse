"""Base forecaster interface wrapping sktime estimators for unified demand forecasting."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Union
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone as sk_clone
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

# Silence sktime 1.3 transition warnings cleanly
warnings.filterwarnings("ignore", message=".*remember_data.*", category=FutureWarning)

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Forecasting.Base")


@dataclass
class ForecastResult:
    """Standardized output container for model predictions and benchmark performance."""

    model_name: str
    series_id: str
    horizon: int
    predictions: List[float]
    actuals: Optional[List[float]] = None
    timestamps: Optional[List[str]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    fit_time_sec: float = 0.0
    predict_time_sec: float = 0.0
    model_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DemandForecasterWrapper:
    """Unified wrapper around sktime forecasting estimators."""

    def __init__(
        self,
        name: str,
        estimator: BaseForecaster,
        clip_non_negative: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initializes wrapper with a sktime BaseForecaster instance.

        Args:
            name: Human-readable model identifier.
            estimator: Underlying sktime forecasting estimator.
            clip_non_negative: If True, clips negative forecasts to zero (physical retail demand constraint).
            params: Dictionary of hyperparameter configurations.
        """
        self.name = name
        self.estimator = estimator
        self.clip_non_negative = clip_non_negative
        self.params = params or {}
        self.is_fitted = False
        self.fit_time_sec = 0.0
        self.predict_time_sec = 0.0
        self._y_train: Optional[pd.Series] = None

    def clone(self) -> "DemandForecasterWrapper":
        """Returns an unfitted, pristine clone of this forecaster wrapper."""
        return DemandForecasterWrapper(
            name=self.name,
            estimator=sk_clone(self.estimator),
            clip_non_negative=self.clip_non_negative,
            params=dict(self.params),
        )

    @staticmethod
    def _ensure_valid_temporal_index(y: pd.Series) -> pd.Series:
        """Enforces a valid, monotonic temporal frequency on the Series index."""
        if isinstance(y.index, pd.DatetimeIndex):
            if y.index.freq is None:
                inferred = pd.infer_freq(y.index)
                if inferred:
                    y = y.asfreq(inferred)
                else:
                    y = y.copy()
                    y.index.freq = "D"
        return y

    def fit(
        self,
        y: pd.Series,
        X: Optional[pd.DataFrame] = None,
        fh: Optional[Union[int, List[int], np.ndarray, ForecastingHorizon]] = None,
    ) -> "DemandForecasterWrapper":
        """Fits the underlying sktime forecaster on historical demand.

        Args:
            y: Historical demand time series with datetime or integer index.
            X: Optional exogenous features.
            fh: Optional forecasting horizon passed at fit time (required by direct reduction).

        Returns:
            self
        """
        y_clean = self._ensure_valid_temporal_index(y)
        self._y_train = y_clean.copy()
        start = time.perf_counter()
        logger.debug(f"Fitting model '{self.name}' on {len(y_clean)} observations...")

        if fh is not None and not isinstance(fh, ForecastingHorizon):
            if isinstance(fh, int):
                fh = ForecastingHorizon(np.arange(1, fh + 1), is_relative=True)
            elif isinstance(fh, (list, np.ndarray)):
                fh = ForecastingHorizon(np.asarray(fh), is_relative=True)

        try:
            self.estimator.fit(y=y_clean, X=X, fh=fh)
        except (TypeError, ValueError) as err:
            err_msg = str(err).lower()
            if X is not None and ("exogenous" in err_msg or "unexpected keyword" in err_msg or "does not support" in err_msg):
                logger.warning(f"Forecaster '{self.name}' does not accept exogenous X ({err}). Refitting without X.")
                self.estimator.fit(y=y_clean, fh=fh)
            else:
                raise
        self.fit_time_sec = round(time.perf_counter() - start, 4)
        self.is_fitted = True
        return self

    def predict(
        self,
        fh: Union[int, List[int], np.ndarray, ForecastingHorizon],
        X: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Generates future point forecasts across the specified forecasting horizon.

        Args:
            fh: Forecasting horizon (e.g. integer 7, or array [1, 2, ..., 7]).
            X: Optional exogenous features for the forecast horizon.

        Returns:
            Pandas Series of predicted demand.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Forecaster '{self.name}' must be fitted before predict() is called.")

        start = time.perf_counter()
        if isinstance(fh, int):
            fh_obj = ForecastingHorizon(np.arange(1, fh + 1), is_relative=True)
        elif isinstance(fh, (list, np.ndarray)):
            fh_obj = ForecastingHorizon(np.asarray(fh), is_relative=True)
        else:
            fh_obj = fh

        try:
            raw_preds = self.estimator.predict(fh=fh_obj, X=X)
        except (TypeError, ValueError) as err:
            err_msg = str(err).lower()
            if X is not None and ("exogenous" in err_msg or "unexpected keyword" in err_msg or "does not support" in err_msg):
                raw_preds = self.estimator.predict(fh=fh_obj)
            else:
                raise
        self.predict_time_sec = round(time.perf_counter() - start, 4)

        if self.clip_non_negative:
            # Physical constraint: retail demand cannot be negative
            raw_preds = np.maximum(0.0, raw_preds)

        if not isinstance(raw_preds, pd.Series):
            raw_preds = pd.Series(raw_preds)

        return raw_preds

    def fit_and_evaluate(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        series_id: str = "series",
        sp: int = 7,
    ) -> ForecastResult:
        """Fits on training data, predicts on test horizon, and computes benchmark evaluation metrics.

        Args:
            y_train: Historical training observations.
            y_test: Future ground-truth test observations (length defines horizon H).
            series_id: Identifier for the time-series.
            sp: Seasonal periodicity for MASE.

        Returns:
            ForecastResult containing predictions, actuals, and calculated metrics.
        """
        horizon = len(y_test)
        fh_obj = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
        self.fit(y=y_train, fh=fh_obj)
        predictions = self.predict(fh=fh_obj)

        preds_array = np.asarray(predictions, dtype=float)
        actuals_array = np.asarray(y_test, dtype=float)

        from src.evaluation.metrics import calculate_forecast_metrics
        metrics = calculate_forecast_metrics(
            y_true=actuals_array,
            y_pred=preds_array,
            y_train=np.asarray(y_train, dtype=float),
            sp=sp,
        )

        timestamps = [str(t) for t in y_test.index] if hasattr(y_test, "index") else None

        return ForecastResult(
            model_name=self.name,
            series_id=series_id,
            horizon=horizon,
            predictions=[round(float(p), 4) for p in preds_array],
            actuals=[round(float(a), 4) for a in actuals_array],
            timestamps=timestamps,
            metrics=metrics,
            fit_time_sec=self.fit_time_sec,
            predict_time_sec=self.predict_time_sec,
            model_config=self.params,
        )

