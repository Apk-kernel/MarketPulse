"""Baseline time-series forecasting models using sktime.

Implements:
1. NaiveForecaster (last known value)
2. NaiveForecaster (mean strategy)
3. Seasonal NaiveForecaster (sp=7, weekly seasonal persistence)
4. ExponentialSmoothing (Holt-Winters additive)
5. ThetaForecaster (Assimakopoulos & Nikolopoulos Theta method)
6. AutoARIMA (pmdarima) with fallback to SARIMAX (statsmodels)
"""

from typing import Any, Dict, List, Optional
import warnings

# sktime estimators
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.croston import Croston
from sktime.forecasting.sarimax import SARIMAX

from src.forecasting.base import DemandForecasterWrapper
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Forecasting.Baselines")


def create_naive_last() -> DemandForecasterWrapper:
    """Creates NaiveForecaster predicting the most recent historical observation."""
    estimator = NaiveForecaster(strategy="last")
    return DemandForecasterWrapper(
        name="Naive_Last",
        estimator=estimator,
        params={"strategy": "last"},
    )


def create_naive_mean() -> DemandForecasterWrapper:
    """Creates NaiveForecaster predicting the global historical mean."""
    estimator = NaiveForecaster(strategy="mean")
    return DemandForecasterWrapper(
        name="Naive_Mean",
        estimator=estimator,
        params={"strategy": "mean"},
    )


def create_seasonal_naive(sp: int = 7) -> DemandForecasterWrapper:
    """Creates Seasonal NaiveForecaster repeating the observation from the previous seasonal cycle.

    In sktime, seasonal naive is achieved via strategy='last' with sp > 1.

    Args:
        sp: Seasonal periodicity (7 for weekly cycle in daily data).
    """
    estimator = NaiveForecaster(strategy="last", sp=sp)
    return DemandForecasterWrapper(
        name=f"Seasonal_Naive_sp{sp}",
        estimator=estimator,
        params={"strategy": "last", "sp": sp},
    )


def create_exponential_smoothing(
    sp: int = 7,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
) -> DemandForecasterWrapper:
    """Creates Holt-Winters ExponentialSmoothing forecaster.

    Args:
        sp: Seasonal periodicity.
        trend: Trend component ('add', 'mul', or None).
        seasonal: Seasonal component ('add', 'mul', or None).
    """
    estimator = ExponentialSmoothing(
        trend=trend,
        seasonal=seasonal,
        sp=sp,
        initialization_method="estimated",
    )
    return DemandForecasterWrapper(
        name="Exponential_Smoothing",
        estimator=estimator,
        params={"trend": trend, "seasonal": seasonal, "sp": sp},
    )


def create_theta_forecaster(sp: int = 7) -> DemandForecasterWrapper:
    """Creates ThetaForecaster (Assimakopoulos and Nikolopoulos, 2000).

    Uses additive deseasonalization model to robustly handle zero-sales periods common in retail demand.

    Args:
        sp: Seasonal periodicity.
    """
    estimator = ThetaForecaster(sp=sp, deseasonalize_model="additive")
    return DemandForecasterWrapper(
        name="Theta",
        estimator=estimator,
        params={"sp": sp, "deseasonalize_model": "additive"},
    )


def create_croston(smoothing: float = 0.1) -> DemandForecasterWrapper:
    """Creates Croston forecaster for intermittent and erratic demand series.

    Args:
        smoothing: Exponential smoothing parameter for demand size and inter-arrival intervals.
    """
    estimator = Croston(smoothing=smoothing)
    return DemandForecasterWrapper(
        name="Croston",
        estimator=estimator,
        params={"smoothing": smoothing},
    )


def create_auto_arima(sp: int = 7) -> DemandForecasterWrapper:
    """Creates AutoARIMA forecaster with pmdarima backend, or fallback to SARIMAX if pmdarima is absent.

    Args:
        sp: Seasonal periodicity (default: 7 for weekly retail seasonality).
    """
    try:
        from sktime.forecasting.arima import AutoARIMA
        estimator = AutoARIMA(
            sp=sp,
            max_p=2,
            max_q=2,
            seasonal=bool(sp > 1),
            suppress_warnings=True,
            stepwise=True,
        )
        return DemandForecasterWrapper(
            name="AutoARIMA",
            estimator=estimator,
            params={"backend": "pmdarima", "sp": sp, "max_p": 2, "max_q": 2},
        )
    except Exception as e:
        logger.warning(
            f"AutoARIMA unavailable ({e}). Falling back to statsmodels SARIMAX(1, 1, 1)."
        )
        estimator = SARIMAX(
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, sp) if sp > 1 else (0, 0, 0, 0),
        )
        return DemandForecasterWrapper(
            name="SARIMAX_Fallback",
            estimator=estimator,
            params={"backend": "statsmodels", "order": (1, 1, 1), "sp": sp},
        )


def build_all_baseline_forecasters(sp: int = 7) -> Dict[str, DemandForecasterWrapper]:
    """Instantiates the complete dictionary of baseline forecasting estimators.

    Args:
        sp: Seasonal periodicity (default: 7 for daily e-commerce demand).

    Returns:
        Dictionary mapping model key to DemandForecasterWrapper instance.
    """
    forecasters = {
        "naive_last": create_naive_last(),
        "naive_mean": create_naive_mean(),
        "seasonal_naive": create_seasonal_naive(sp=sp),
        "exponential_smoothing": create_exponential_smoothing(sp=sp),
        "theta": create_theta_forecaster(sp=sp),
        "auto_arima": create_auto_arima(sp=sp),
    }
    return forecasters

