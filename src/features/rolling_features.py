"""Rolling window feature generation with strict lag shifts to prevent lookahead leakage."""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Features.Rolling")


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: Optional[List[str]] = None,
    mean_windows: Optional[List[int]] = None,
    std_windows: Optional[List[int]] = None,
    shift: int = 1,
) -> pd.DataFrame:
    """Generates rolling window summary statistics strictly shifted to prevent lookahead leakage.

    Zero Leakage Formulation:
        Rolling statistics for observation at time t are calculated strictly over the window:
            [t - shift - window + 1, ..., t - shift]
        By default, shift=1 ensures y_t is never included in features used to forecast y_t.

    Args:
        df: DataFrame containing demand time series.
        target_col: Target variable name.
        group_cols: Grouping columns for multi-series panels.
        mean_windows: Window sizes for rolling means (defaults to [7, 14, 28]).
        std_windows: Window sizes for rolling standard deviations (defaults to [7, 28]).
        shift: Number of periods to shift backward before computing window stats (must be >= 1).

    Returns:
        DataFrame augmented with rolling mean and std features.
    """
    if shift < 1:
        raise ValueError(f"Shift must be >= 1 to prevent lookahead target leakage, got {shift}")

    df = df.copy()
    if mean_windows is None:
        mean_windows = [7, 14, 28]
    if std_windows is None:
        std_windows = [7, 28]

    # Shift target first
    if group_cols:
        shifted_target = df.groupby(group_cols, observed=True)[target_col].shift(shift)
    else:
        shifted_target = df[target_col].shift(shift)

    # 1. Rolling Means
    for w in mean_windows:
        col_name = f"rolling_mean_{w}"
        if group_cols:
            df[col_name] = (
                df.groupby(group_cols, observed=True)[target_col]
                .transform(lambda s: s.shift(shift).rolling(window=w, min_periods=1).mean())
                .astype("float32")
            )
        else:
            df[col_name] = (
                shifted_target.rolling(window=w, min_periods=1)
                .mean()
                .astype("float32")
            )

    # 2. Rolling Standard Deviations
    for w in std_windows:
        col_name = f"rolling_std_{w}"
        if group_cols:
            df[col_name] = (
                df.groupby(group_cols, observed=True)[target_col]
                .transform(lambda s: s.shift(shift).rolling(window=w, min_periods=2).std())
                .fillna(0.0)
                .astype("float32")
            )
        else:
            df[col_name] = (
                shifted_target.rolling(window=w, min_periods=2)
                .std()
                .fillna(0.0)
                .astype("float32")
            )

    logger.debug(
        f"Generated rolling features: means={mean_windows}, stds={std_windows} with shift={shift}"
    )
    return df


def create_price_dynamics_features(
    df: pd.DataFrame,
    price_col: str = "sell_price",
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Computes price dynamics: previous price, price difference, and rolling average price.

    Args:
        df: DataFrame with price column.
        price_col: Name of price column.
        group_cols: Grouping identifiers (e.g. ['store_id', 'item_id']).

    Returns:
        DataFrame augmented with price dynamics features.
    """
    df = df.copy()
    if price_col not in df.columns:
        return df

    if group_cols:
        grouped = df.groupby(group_cols, observed=True)[price_col]
        df["prev_price"] = grouped.shift(1).astype("float32")
        df["price_diff"] = (df[price_col] - df["prev_price"]).fillna(0.0).astype("float32")
        df["price_pct_change"] = np.where(
            df["prev_price"] > 0,
            (df[price_col] - df["prev_price"]) / df["prev_price"],
            0.0,
        ).astype("float32")
        df["rolling_price_mean_7"] = (
            grouped.transform(lambda s: s.rolling(window=7, min_periods=1).mean())
            .astype("float32")
        )
        df["rolling_price_mean_28"] = (
            grouped.transform(lambda s: s.rolling(window=28, min_periods=1).mean())
            .astype("float32")
        )
    else:
        df["prev_price"] = df[price_col].shift(1).astype("float32")
        df["price_diff"] = (df[price_col] - df["prev_price"]).fillna(0.0).astype("float32")
        df["price_pct_change"] = np.where(
            df["prev_price"] > 0,
            (df[price_col] - df["prev_price"]) / df["prev_price"],
            0.0,
        ).astype("float32")
        df["rolling_price_mean_7"] = (
            df[price_col].rolling(window=7, min_periods=1).mean().astype("float32")
        )
        df["rolling_price_mean_28"] = (
            df[price_col].rolling(window=28, min_periods=1).mean().astype("float32")
        )

    return df

