"""Price dynamics, discount depth, and empirical price elasticity estimation."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.PriceFeatures")


def calculate_price_features(
    df: pd.DataFrame,
    item_col: str = "item_id",
    store_col: str = "store_id",
    price_col: str = "sell_price",
    date_col: Optional[str] = "date",
    use_expanding_max: bool = True,
) -> pd.DataFrame:
    """Computes price statistics, discount depth, and promotional indicators.

    Ensures zero temporal lookahead leakage by calculating the historical base price
    as the causal expanding maximum price up to time t when a date column is present.

    Args:
        df: Merged demand DataFrame with sell_price.
        item_col: Item identifier column.
        store_col: Store identifier column.
        price_col: Sell price column.
        date_col: Date column name for chronological ordering (default: 'date').
        use_expanding_max: If True and date_col is present, uses causal expanding cummax
            to strictly prevent future price leakage into historical base prices.

    Returns:
        DataFrame augmented with price discount features.
    """
    df = df.copy()
    if price_col not in df.columns:
        logger.warning(f"Price column '{price_col}' not found.")
        return df

    # Calculate historical base price per (item, store)
    if use_expanding_max and date_col and date_col in df.columns:
        orig_idx = df.index
        sort_keys = (
            pd.to_datetime(df[date_col])
            if not pd.api.types.is_datetime64_any_dtype(df[date_col])
            else df[date_col]
        )
        sorted_indices = sort_keys.argsort(kind="mergesort")
        sorted_df = df.iloc[sorted_indices]
        cummax_series = sorted_df.groupby([store_col, item_col], observed=True)[price_col].cummax()
        max_price = cummax_series.reindex(orig_idx)
    else:
        max_price = df.groupby([store_col, item_col], observed=True)[price_col].transform("max")

    df["base_price"] = max_price
    # Discount depth: 0 = regular price, 0.20 = 20% discount
    df["discount_depth"] = np.where(
        max_price > 0,
        np.maximum(0.0, (max_price - df[price_col]) / max_price),
        0.0,
    ).astype("float32")
    df["is_discounted"] = (df["discount_depth"] > 0.05).astype("int8")

    return df


def estimate_price_elasticity(
    df: pd.DataFrame,
    sales_col: str = "sales",
    price_col: str = "sell_price",
    group_col: Optional[str] = "cat_id",
) -> pd.DataFrame:
    """Estimates empirical price elasticity of demand via log-log regression:

        ln(Sales + 1) = alpha + beta * ln(Price) + error

    where beta is the price elasticity of demand.

    Args:
        df: Merged demand DataFrame.
        sales_col: Sales column name.
        price_col: Price column name.
        group_col: Optional grouping column (e.g. 'cat_id', 'dept_id', or None for overall).

    Returns:
        DataFrame containing elasticity coefficients, p-values, and R-squared.
    """
    results = []

    def _regress(sub_df: pd.DataFrame, label: str) -> Dict[str, object]:
        valid = sub_df[[sales_col, price_col]].dropna()
        valid = valid[valid[price_col] > 0]
        if len(valid) < 10 or valid[price_col].nunique() < 2:
            return {
                "group": label,
                "elasticity": np.nan,
                "r_squared": np.nan,
                "p_value": np.nan,
                "sample_size": len(valid),
                "interpretation": "Insufficient price variation",
            }

        log_p = np.log(valid[price_col].values)
        log_q = np.log(valid[sales_col].values + 1.0)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_p, log_q)
        r2 = r_value**2

        if np.isnan(slope):
            interp = "Indeterminate"
        elif slope < -1.0:
            interp = "Elastic (High promotional sensitivity)"
        elif -1.0 <= slope < 0.0:
            interp = "Inelastic (Staple demand)"
        else:
            interp = "Positive slope (Price-insensitive / Luxury anomaly)"

        return {
            "group": label,
            "elasticity": round(float(slope), 4),
            "r_squared": round(float(r2), 4),
            "p_value": round(float(p_value), 6),
            "sample_size": len(valid),
            "interpretation": interp,
        }

    if group_col and group_col in df.columns:
        for grp, grp_df in df.groupby(group_col, observed=True):
            results.append(_regress(grp_df, str(grp)))
    else:
        results.append(_regress(df, "Overall"))

    return pd.DataFrame(results)

