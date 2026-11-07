"""Global and Local Machine Learning Forecasting using sktime's reduction framework.

Implements:
1. Local Tabular Reduction (independently trained models per series)
2. Global Tabular Reduction (shared models pooled across series)
3. Recursive Reduction vs Direct Multi-step Reduction
4. Candidate Regressors: LightGBM, HistGradientBoosting, RandomForest
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from sktime.forecasting.compose import make_reduction

from src.forecasting.base import DemandForecasterWrapper
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Forecasting.GlobalModels")


def get_base_regressor(
    name: str = "lightgbm",
    random_state: int = 42,
    **kwargs,
) -> BaseEstimator:
    """Instantiates a supervised regression model configured for demand forecasting.

    Args:
        name: Regressor type ('lightgbm', 'hist_gradient_boosting', 'random_forest').
        random_state: Reproducibility seed.
        **kwargs: Additional hyperparameter overrides.

    Returns:
        Scikit-learn compatible regressor instance.
    """
    key = name.lower()
    if key in ["lightgbm", "lgb", "lgbm"]:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return LGBMRegressor(**params)

    elif key in ["hist_gradient_boosting", "histgb", "hgb"]:
        params = {
            "max_iter": 100,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "random_state": random_state,
        }
        params.update(kwargs)
        return HistGradientBoostingRegressor(**params)

    elif key in ["random_forest", "rf"]:
        params = {
            "n_estimators": 80,
            "max_depth": 12,
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return RandomForestRegressor(**params)

    else:
        raise ValueError(
            f"Unsupported regressor '{name}'. Expected 'lightgbm', 'hist_gradient_boosting', or 'random_forest'."
        )


def create_reduction_forecaster(
    regressor_name: str = "lightgbm",
    strategy: str = "recursive",
    window_length: int = 28,
    pooling: str = "local",
    random_state: int = 42,
    **regressor_kwargs,
) -> DemandForecasterWrapper:
    """Constructs a reduction forecaster wrapping tabular ML models inside sktime.

    Args:
        regressor_name: Candidate regressor ('lightgbm', 'hist_gradient_boosting', 'random_forest').
        strategy: Multi-step forecasting strategy ('recursive' or 'direct').
        window_length: Number of past demand lag observations used as input window.
        pooling: 'local' (separate model per series) or 'global' (shared model across series).
        random_state: Random seed.
        **regressor_kwargs: Additional arguments for the underlying regressor.

    Returns:
        DemandForecasterWrapper instance.
    """
    regressor = get_base_regressor(
        name=regressor_name,
        random_state=random_state,
        **regressor_kwargs,
    )

    reduction_estimator = make_reduction(
        estimator=regressor,
        strategy=strategy,
        window_length=window_length,
        pooling=pooling,
        scitype="tabular-regressor",
    )

    wrapper_name = f"{pooling.capitalize()}_{regressor_name}_{strategy}_w{window_length}"

    return DemandForecasterWrapper(
        name=wrapper_name,
        estimator=reduction_estimator,
        clip_non_negative=True,
        params={
            "regressor": regressor_name,
            "strategy": strategy,
            "window_length": window_length,
            "pooling": pooling,
            "random_state": random_state,
        },
    )


def build_ml_reduction_models(
    window_length: int = 28,
    random_state: int = 42,
) -> Dict[str, DemandForecasterWrapper]:
    """Builds standard suite of ML reduction models comparing regressors and recursive vs direct strategies.

    Args:
        window_length: Input lag window (default: 28 days for monthly retail cycle).
        random_state: Reproducibility seed.

    Returns:
        Dictionary of candidate ML forecasters.
    """
    models = {
        "lgb_recursive": create_reduction_forecaster(
            regressor_name="lightgbm",
            strategy="recursive",
            window_length=window_length,
            pooling="local",
            random_state=random_state,
        ),
        "lgb_direct": create_reduction_forecaster(
            regressor_name="lightgbm",
            strategy="direct",
            window_length=window_length,
            pooling="local",
            random_state=random_state,
        ),
        "hgb_recursive": create_reduction_forecaster(
            regressor_name="hist_gradient_boosting",
            strategy="recursive",
            window_length=window_length,
            pooling="local",
            random_state=random_state,
        ),
        "rf_recursive": create_reduction_forecaster(
            regressor_name="random_forest",
            strategy="recursive",
            window_length=window_length,
            pooling="local",
            random_state=random_state,
        ),
    }
    return models


def prepare_global_panel_dataframe(
    series_dict: Dict[str, pd.Series],
    target_col: str = "sales",
) -> pd.DataFrame:
    """Converts a dictionary of univariate series into a multi-index panel DataFrame for global reduction.

    Format required by sktime: MultiIndex [instances, time_index].

    Args:
        series_dict: Dict mapping series_id to univariate pd.Series with DatetimeIndex.
        target_col: Name for target column.

    Returns:
        MultiIndex DataFrame suitable for sktime pooling='global'.
    """
    dfs = []
    for s_id, s_series in series_dict.items():
        s_clean = DemandForecasterWrapper._ensure_valid_temporal_index(s_series)
        sub_df = pd.DataFrame({target_col: s_clean.values}, index=s_clean.index)
        sub_df["instance"] = s_id
        dfs.append(sub_df)

    combined = pd.concat(dfs).reset_index()
    time_col = combined.columns[0]
    combined = combined.set_index(["instance", time_col]).sort_index()
    return combined

