"""Lag feature generation with strict temporal ordering and zero lookahead leakage."""

from typing import List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Features.Lags")


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Generates autoregressive lag features for univariate or grouped multi-series demand.

    Zero Leakage Guarantee:
        For target observation y_t, lag_k feature contains strictly y_{t-k} (k >= 1).
        Future values y_{t+j} are never accessed.

    Args:
        df: DataFrame containing target and grouping columns.
        target_col: Column name of demand target.
        group_cols: Optional grouping columns for panel/hierarchical series (e.g. ['store_id', 'item_id']).
        lags: List of integer lags to compute. Defaults to [1, 7, 14, 28, 56].

    Returns:
        DataFrame augmented with lag_{k} columns.
    """
    df = df.copy()
    if lags is None:
        lags = [1, 7, 14, 28, 56]

    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1 to prevent target leakage, got {lag}")

        col_name = f"lag_{lag}"
        if group_cols:
            df[col_name] = (
                df.groupby(group_cols, observed=True)[target_col]
                .shift(lag)
                .astype("float32")
            )
        else:
            df[col_name] = df[target_col].shift(lag).astype("float32")

    logger.debug(f"Generated {len(lags)} lag features: {[f'lag_{l}' for l in lags]}")
    return df

