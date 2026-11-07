"""Demand characterization and Syntetos-Boylan demand regime classification."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.DemandFeatures")


@dataclass
class DemandProfile:
    """Statistical summary and regime categorization for a demand series."""

    series_id: str
    total_periods: int
    active_periods: int
    mean_sales: float
    median_sales: float
    std_sales: float
    zero_ratio: float
    non_zero_mean: float
    non_zero_std: float
    adi: float  # Average Demand Interval
    cv2: float  # Squared Coefficient of Variation
    regime: str  # 'Smooth', 'Intermittent', 'Erratic', 'Lumpy'
    autocorr_lag1: float
    autocorr_lag7: float
    autocorr_lag28: float

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return asdict(self)


def calculate_demand_profile(
    series: pd.Series,
    series_id: str = "series",
    adi_cutoff: float = 1.32,
    cv2_cutoff: float = 0.49,
) -> DemandProfile:
    """Calculates comprehensive demand diagnostics and classifies into Syntetos-Boylan regimes.

    Mathematical Formulation:
        - ADI = Total Periods / Number of Non-Zero Periods
        - CV^2 = (Std(Non-Zero Demand) / Mean(Non-Zero Demand))^2

    Regimes:
        - Smooth: ADI < 1.32 and CV^2 < 0.49
        - Intermittent: ADI >= 1.32 and CV^2 < 0.49
        - Erratic: ADI < 1.32 and CV^2 >= 0.49
        - Lumpy: ADI >= 1.32 and CV^2 >= 0.49

    Args:
        series: Pandas series of sales demand (active observations).
        series_id: Identifier for the series.
        adi_cutoff: ADI threshold (default: 1.32).
        cv2_cutoff: CV^2 threshold (default: 0.49).

    Returns:
        DemandProfile dataclass.
    """
    clean_series = series.dropna().astype(float)
    n = len(clean_series)

    if n == 0:
        return DemandProfile(
            series_id=series_id,
            total_periods=0,
            active_periods=0,
            mean_sales=0.0,
            median_sales=0.0,
            std_sales=0.0,
            zero_ratio=1.0,
            non_zero_mean=0.0,
            non_zero_std=0.0,
            adi=np.nan,
            cv2=np.nan,
            regime="Unknown",
            autocorr_lag1=0.0,
            autocorr_lag7=0.0,
            autocorr_lag28=0.0,
        )

    non_zero = clean_series[clean_series > 0]
    n_nonzero = len(non_zero)
    zero_ratio = (n - n_nonzero) / n

    mean_s = float(clean_series.mean())
    median_s = float(clean_series.median())
    std_s = float(clean_series.std()) if n > 1 else 0.0

    if n_nonzero > 0:
        nz_mean = float(non_zero.mean())
        nz_std = float(non_zero.std()) if n_nonzero > 1 else 0.0
        # ADI: Average interval between positive demand occurrences
        adi = float(n / n_nonzero)
        # CV^2 of non-zero demand size
        cv2 = float((nz_std / nz_mean) ** 2) if nz_mean > 0 else 0.0
    else:
        nz_mean = 0.0
        nz_std = 0.0
        adi = float(n)  # Infinite interval in practical terms
        cv2 = 0.0

    # Categorization
    if n_nonzero == 0:
        regime = "ZeroDemand"
    elif adi < adi_cutoff and cv2 < cv2_cutoff:
        regime = "Smooth"
    elif adi >= adi_cutoff and cv2 < cv2_cutoff:
        regime = "Intermittent"
    elif adi < adi_cutoff and cv2 >= cv2_cutoff:
        regime = "Erratic"
    else:
        regime = "Lumpy"

    # Autocorrelations with length and variation guards
    ac1 = float(clean_series.autocorr(lag=1)) if n > 3 and std_s > 1e-6 else 0.0
    ac7 = float(clean_series.autocorr(lag=7)) if n > 14 and std_s > 1e-6 else 0.0
    ac28 = float(clean_series.autocorr(lag=28)) if n > 35 and std_s > 1e-6 else 0.0

    # Replace NaNs in autocorrelations (e.g. constant series)
    ac1 = 0.0 if np.isnan(ac1) else round(ac1, 4)
    ac7 = 0.0 if np.isnan(ac7) else round(ac7, 4)
    ac28 = 0.0 if np.isnan(ac28) else round(ac28, 4)

    return DemandProfile(
        series_id=series_id,
        total_periods=n,
        active_periods=n_nonzero,
        mean_sales=round(mean_s, 4),
        median_sales=round(median_s, 4),
        std_sales=round(std_s, 4),
        zero_ratio=round(zero_ratio, 4),
        non_zero_mean=round(nz_mean, 4),
        non_zero_std=round(nz_std, 4),
        adi=round(adi, 4),
        cv2=round(cv2, 4),
        regime=regime,
        autocorr_lag1=ac1,
        autocorr_lag7=ac7,
        autocorr_lag28=ac28,
    )


def classify_demand_regimes(
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    sales_col: str = "sales",
    adi_cutoff: float = 1.32,
    cv2_cutoff: float = 0.49,
) -> pd.DataFrame:
    """Performs demand regime profiling across all series in a long-format DataFrame.

    Args:
        df: Long-format demand DataFrame.
        id_cols: Columns uniquely identifying a series (e.g. ['store_id', 'item_id']).
        sales_col: Sales column name.
        adi_cutoff: ADI threshold.
        cv2_cutoff: CV^2 threshold.

    Returns:
        DataFrame containing one row per series with full DemandProfile metrics.
    """
    if id_cols is None:
        if "id" in df.columns:
            id_cols = ["id"]
        else:
            id_cols = ["store_id", "item_id"]

    profiles = []
    # Use observed=True to handle categorical index
    grouped = df.groupby(id_cols, observed=True)

    for group_key, group_data in grouped:
        if isinstance(group_key, tuple):
            series_id = "_".join(str(k) for k in group_key)
        else:
            series_id = str(group_key)

        sales_series = group_data[sales_col]
        # Only analyze active periods if is_active column exists
        if "is_active" in group_data.columns:
            sales_series = sales_series[group_data["is_active"]]

        prof = calculate_demand_profile(
            sales_series,
            series_id=series_id,
            adi_cutoff=adi_cutoff,
            cv2_cutoff=cv2_cutoff,
        )
        rec = prof.to_dict()
        if isinstance(group_key, tuple):
            for col, val in zip(id_cols, group_key):
                rec[col] = val
        else:
            rec[id_cols[0]] = group_key
        profiles.append(rec)

    result_df = pd.DataFrame(profiles)
    logger.info(
        f"Classified {len(result_df)} series into demand regimes: "
        f"{result_df['regime'].value_counts().to_dict()}"
    )
    return result_df

