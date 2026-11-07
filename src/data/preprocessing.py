"""Data Preprocessing Pipeline: Wide-to-Long transformation, Merging, and Cleaning."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.validation import DataValidator, DataQualityReport
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Preprocessing")


class M5Preprocessor:
    """End-to-end preprocessing pipeline for the M5 Walmart demand dataset."""

    def __init__(
        self,
        downcast_dtypes: bool = True,
        flag_unlisted_as_nan: bool = False,
    ):
        """Initializes preprocessor.

        Args:
            downcast_dtypes: Whether to convert string IDs to categories and downcast numeric types.
            flag_unlisted_as_nan: If True, replaces sales with NaN for dates prior to an item's launch
                                 (identified by missing sell_price). If False, preserves sales but
                                 adds an explicit `is_active` indicator.
        """
        self.downcast_dtypes = downcast_dtypes
        self.flag_unlisted_as_nan = flag_unlisted_as_nan

    def melt_sales_wide_to_long(
        self,
        sales_df: pd.DataFrame,
        id_vars: Optional[List[str]] = None,
        day_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Transforms wide sales matrix into long format.

        Args:
            sales_df: Raw sales dataframe with 'd_1', 'd_2', ...
            id_vars: Identifier columns to preserve.
            day_cols: Specific day columns to melt. If None, all 'd_*' columns are used.

        Returns:
            Long-format sales DataFrame with columns [id_vars..., 'd', 'sales'].
        """
        if id_vars is None:
            id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        # Validate that id_vars exist
        existing_id_vars = [c for c in id_vars if c in sales_df.columns]

        if day_cols is None:
            day_cols = [c for c in sales_df.columns if c.startswith("d_")]

        logger.info(
            f"Melting sales dataframe: {len(sales_df):,} series across {len(day_cols):,} days."
        )

        melted = pd.melt(
            sales_df,
            id_vars=existing_id_vars,
            value_vars=day_cols,
            var_name="d",
            value_name="sales",
        )

        # Convert sales to float32 to support possible missing/unlisted states
        melted["sales"] = melted["sales"].astype("float32")
        return melted

    def merge_calendar_and_prices(
        self,
        melted_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merges calendar features and sell prices onto the long-format sales dataframe.

        Args:
            melted_df: Output from melt_sales_wide_to_long.
            calendar_df: Calendar dataframe.
            prices_df: Sell prices dataframe.

        Returns:
            Merged and enriched DataFrame.
        """
        logger.info("Merging calendar attributes onto sales.")
        cal_cols = [
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
        available_cal_cols = [c for c in cal_cols if c in calendar_df.columns]
        merged = melted_df.merge(calendar_df[available_cal_cols], on="d", how="left")

        # Standardize state-specific SNAP benefits into a single store_snap column
        if "state_id" in merged.columns:
            snap_col = pd.Series(0, index=merged.index, dtype="int8")
            if "snap_CA" in merged.columns:
                snap_col = snap_col.where(merged["state_id"] != "CA", merged["snap_CA"])
            if "snap_TX" in merged.columns:
                snap_col = snap_col.where(merged["state_id"] != "TX", merged["snap_TX"])
            if "snap_WI" in merged.columns:
                snap_col = snap_col.where(merged["state_id"] != "WI", merged["snap_WI"])
            merged["snap_active"] = snap_col

        logger.info("Merging sell prices onto sales.")
        # Ensure join keys have matching string types
        merged["store_id"] = merged["store_id"].astype(str)
        merged["item_id"] = merged["item_id"].astype(str)
        prices_clean = prices_df.copy()
        prices_clean["store_id"] = prices_clean["store_id"].astype(str)
        prices_clean["item_id"] = prices_clean["item_id"].astype(str)

        merged = merged.merge(
            prices_clean[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        # Distinguish unlisted items from active items:
        # In M5, an item with no sell_price in that week is unlisted in that store.
        merged["is_active"] = merged["sell_price"].notna()

        if self.flag_unlisted_as_nan:
            # Mask pre-launch sales as NaN so they do not falsely skew zero-sales statistics
            logger.info("Flagging unlisted item sales as NaN based on missing prices.")
            merged.loc[~merged["is_active"], "sales"] = np.nan

        return merged

    def clean_and_optimize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimizes dtypes for memory efficiency and standardizes categorical columns.

        Args:
            df: Merged dataframe.

        Returns:
            Memory-optimized DataFrame.
        """
        logger.info("Optimizing column data types.")
        df = df.copy()

        # Date conversion
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # Categorical columns
        cat_cols = [
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "weekday",
            "event_name_1",
            "event_type_1",
        ]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Numerical downcasting
        if "month" in df.columns:
            df["month"] = df["month"].astype("int8")
        if "year" in df.columns:
            df["year"] = df["year"].astype("int16")
        if "wm_yr_wk" in df.columns:
            df["wm_yr_wk"] = df["wm_yr_wk"].astype("int16")
        if "sell_price" in df.columns:
            df["sell_price"] = df["sell_price"].astype("float32")
        if "sales" in df.columns:
            df["sales"] = df["sales"].astype("float32")
        if "snap_active" in df.columns:
            df["snap_active"] = df["snap_active"].astype("int8")
        if "is_active" in df.columns:
            df["is_active"] = df["is_active"].astype("bool")

        # Sort temporally per series for strict time-series ordering
        sort_keys = [c for c in ["store_id", "item_id", "date"] if c in df.columns]
        df = df.sort_values(by=sort_keys).reset_index(drop=True)

        return df

    def process(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        day_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Executes full preprocessing pipeline: melt, merge, clean, and validate.

        Args:
            sales_df: Raw wide sales.
            calendar_df: Raw calendar.
            prices_df: Raw prices.
            day_cols: Optional subset of days to process.

        Returns:
            Tuple of (processed_df, DataQualityReport).
        """
        logger.info("Starting M5 preprocessing pipeline...")
        # Step 1: Schema validation
        DataValidator.validate_raw_sales_schema(sales_df)
        DataValidator.validate_calendar_schema(calendar_df)
        DataValidator.validate_prices_schema(prices_df)

        # Step 2: Melt
        melted = self.melt_sales_wide_to_long(sales_df, day_cols=day_cols)

        # Step 3: Merge calendar and prices
        merged = self.merge_calendar_and_prices(melted, calendar_df, prices_df)

        # Step 4: Clean and optimize dtypes
        processed = self.clean_and_optimize_types(merged)

        # Step 5: Audit processed dataset
        report = DataValidator.audit_processed_dataframe(processed)

        logger.info(
            f"Preprocessing finished: {len(processed):,} rows processed. "
            f"Quality status: {'PASSED' if report.is_valid else 'FAILED'}."
        )
        return processed, report

    @staticmethod
    def save_processed(
        df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = "parquet",
    ) -> Path:
        """Saves processed dataframe to Parquet or CSV.

        Args:
            df: Processed dataframe.
            output_path: Destination file path.
            format: 'parquet' or 'csv'.

        Returns:
            Path of saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "parquet":
            df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        elif format.lower() == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'.")

        logger.info(f"Saved processed dataset ({len(df):,} rows) to {path}")
        return path

