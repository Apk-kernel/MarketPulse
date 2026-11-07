"""Temporal and calendar feature engineering and seasonality analysis."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.TemporalFeatures")


def extract_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
    add_cyclical: bool = True,
) -> pd.DataFrame:
    """Extracts rich calendar features and cyclical sine/cosine transformations.

    Args:
        df: DataFrame containing date column.
        date_col: Name of date column.
        add_cyclical: If True, adds sin/cos transformations for day-of-week and month.

    Returns:
        DataFrame augmented with temporal attributes.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    df["dayofweek"] = dates.dt.dayofweek.astype("int8")  # Monday=0, Sunday=6
    df["dayofmonth"] = dates.dt.day.astype("int8")
    df["month"] = dates.dt.month.astype("int8")
    df["year"] = dates.dt.year.astype("int16")
    df["quarter"] = dates.dt.quarter.astype("int8")
    df["weekofyear"] = dates.dt.isocalendar().week.astype("int8")
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype("int8")
    df["is_month_start"] = dates.dt.is_month_start.astype("int8")
    df["is_month_end"] = dates.dt.is_month_end.astype("int8")

    if add_cyclical:
        # Cyclical day-of-week (period 7)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0).astype("float32")
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0).astype("float32")
        # Cyclical month (period 12)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0).astype("float32")
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0).astype("float32")

    return df


def calculate_seasonal_indices(
    df: pd.DataFrame,
    date_col: str = "date",
    sales_col: str = "sales",
) -> Dict[str, pd.DataFrame]:
    """Calculates empirical seasonal indices: Day-of-Week and Month-of-Year.

    The index is normalized such that an index of 1.0 represents average expected demand.

    Args:
        df: Demand DataFrame.
        date_col: Date column name.
        sales_col: Sales column name.

    Returns:
        Dictionary with 'dow_profile' and 'monthly_profile' DataFrames.
    """
    df_temp = extract_calendar_features(df, date_col=date_col, add_cyclical=False)
    overall_mean = df_temp[sales_col].mean()

    # Day of week
    dow_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    dow_agg = (
        df_temp.groupby("dayofweek")[sales_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    dow_agg["day_name"] = dow_agg["dayofweek"].map(dow_map)
    dow_agg["seasonal_index"] = dow_agg["mean"] / overall_mean if overall_mean > 0 else 1.0

    # Monthly profile
    month_agg = (
        df_temp.groupby("month")[sales_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    month_agg["seasonal_index"] = month_agg["mean"] / overall_mean if overall_mean > 0 else 1.0

    return {
        "dow_profile": dow_agg,
        "monthly_profile": month_agg,
    }


def calculate_event_uplift(
    df: pd.DataFrame,
    event_col: str = "event_name_1",
    sales_col: str = "sales",
    snap_col: str = "snap_active",
) -> Dict[str, object]:
    """Measures statistical uplift during promotional events, holidays, and SNAP periods.

    Args:
        df: Merged demand DataFrame.
        event_col: Event name column.
        sales_col: Sales column.
        snap_col: SNAP active indicator.

    Returns:
        Dictionary with event uplift table and SNAP lift ratio.
    """
    results: Dict[str, object] = {}

    # Event analysis
    if event_col in df.columns:
        has_event = df[event_col].notna() & (df[event_col].astype(str) != "None")
        base_mean = df.loc[~has_event, sales_col].mean()

        event_stats = []
        unique_events = [e for e in df[event_col].dropna().unique() if str(e) != "None"]
        for ev in unique_events:
            ev_mask = df[event_col] == ev
            ev_mean = df.loc[ev_mask, sales_col].mean()
            uplift_ratio = (ev_mean / base_mean) if base_mean > 0 else 1.0
            event_stats.append({
                "event_name": str(ev),
                "count": int(ev_mask.sum()),
                "mean_sales": float(ev_mean),
                "baseline_mean_sales": float(base_mean),
                "uplift_ratio": float(round(uplift_ratio, 4)),
                "uplift_pct": float(round((uplift_ratio - 1.0) * 100, 2)),
            })
        results["events"] = pd.DataFrame(event_stats)
    else:
        results["events"] = pd.DataFrame()

    # SNAP analysis
    if snap_col in df.columns:
        snap_on_mean = df.loc[df[snap_col] == 1, sales_col].mean()
        snap_off_mean = df.loc[df[snap_col] == 0, sales_col].mean()
        snap_lift = (snap_on_mean / snap_off_mean) if snap_off_mean > 0 else 1.0
        results["snap"] = {
            "snap_on_mean": float(round(snap_on_mean, 4)),
            "snap_off_mean": float(round(snap_off_mean, 4)),
            "snap_uplift_ratio": float(round(snap_lift, 4)),
            "snap_uplift_pct": float(round((snap_lift - 1.0) * 100, 2)),
        }
    else:
        results["snap"] = {}

    return results

