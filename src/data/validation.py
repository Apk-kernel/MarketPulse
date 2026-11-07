"""Data Quality Validation and Reporting for MarketPulse."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Validation")


@dataclass
class DataQualityReport:
    """Structured data quality audit report."""

    dataset_name: str
    total_records: int = 0
    unique_series: int = 0
    unique_items: int = 0
    unique_stores: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    total_days: int = 0
    zero_sales_count: int = 0
    zero_sales_pct: float = 0.0
    missing_sales_count: int = 0
    missing_sales_pct: float = 0.0
    negative_sales_count: int = 0
    missing_price_count: int = 0
    missing_price_pct: float = 0.0
    unlisted_item_days_count: int = 0
    duplicate_records_count: int = 0
    date_continuity_violations: int = 0
    hierarchy_violations: int = 0
    is_valid: bool = True
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        """Saves report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved Data Quality Report to {path}")

    def to_markdown(self) -> str:
        """Generates a GitHub-flavored Markdown audit summary."""
        status = "PASSED" if self.is_valid else "FAILED"
        md = [
            f"# Data Quality Report: {self.dataset_name}",
            f"**Validation Status**: `{status}`",
            "",
            "## Summary Statistics",
            f"- **Total Rows**: {self.total_records:,}",
            f"- **Unique Series (Item x Store)**: {self.unique_series:,}",
            f"- **Unique Items**: {self.unique_items:,}",
            f"- **Unique Stores**: {self.unique_stores:,}",
            f"- **Date Range**: {self.start_date} to {self.end_date} ({self.total_days} days)",
            "",
            "## Demand & Integrity Diagnostics",
            f"- **Zero-Sales Observations**: {self.zero_sales_count:,} ({self.zero_sales_pct:.2f}%)",
            f"- **Missing Sales (NaNs)**: {self.missing_sales_count:,} ({self.missing_sales_pct:.2f}%)",
            f"- **Negative Sales**: {self.negative_sales_count:,}",
            f"- **Missing Prices (active items)**: {self.missing_price_count:,} ({self.missing_price_pct:.2f}%)",
            f"- **Unlisted Observations (pre-launch)**: {self.unlisted_item_days_count:,}",
            f"- **Duplicate Keys**: {self.duplicate_records_count:,}",
            f"- **Date Continuity Gaps**: {self.date_continuity_violations:,}",
            f"- **Hierarchy Inconsistencies**: {self.hierarchy_violations:,}",
            "",
        ]
        if self.issues:
            md.append("## Issues Identified")
            for issue in self.issues:
                md.append(f"- ⚠️ {issue}")
        else:
            md.append("## Issues Identified\n- No data quality anomalies detected.")

        return "\n".join(md)


class DataValidator:
    """Enforces statistical and structural validity on raw and processed M5 data."""

    REQUIRED_RAW_SALES_COLS = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
    REQUIRED_CALENDAR_COLS = {"date", "wm_yr_wk", "weekday", "wday", "month", "year", "d"}
    REQUIRED_PRICES_COLS = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    REQUIRED_PROCESSED_COLS = {
        "date",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "sales",
        "sell_price",
        "weekday",
        "month",
        "year",
    }

    @classmethod
    def validate_raw_sales_schema(cls, df: pd.DataFrame) -> None:
        """Validates that raw sales dataframe has all required M5 metadata columns and d_ columns."""
        missing = cls.REQUIRED_RAW_SALES_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Raw sales missing required columns: {sorted(missing)}")

        d_cols = [c for c in df.columns if c.startswith("d_")]
        if not d_cols:
            raise ValueError("Raw sales has no day columns starting with 'd_'")

    @classmethod
    def validate_calendar_schema(cls, df: pd.DataFrame) -> None:
        """Validates calendar dataframe schema."""
        missing = cls.REQUIRED_CALENDAR_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Calendar missing required columns: {sorted(missing)}")

    @classmethod
    def validate_prices_schema(cls, df: pd.DataFrame) -> None:
        """Validates sell prices dataframe schema."""
        missing = cls.REQUIRED_PRICES_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Prices missing required columns: {sorted(missing)}")

    @classmethod
    def audit_processed_dataframe(
        cls,
        df: pd.DataFrame,
        allow_negative_sales: bool = False,
        enforce_continuity: bool = True,
    ) -> DataQualityReport:
        """Performs rigorous quality audit on processed long-format demand data.

        Distinguishes between:
        1. Genuine Zero Sales (item active & on sale, demand = 0).
        2. Missing Observations (explicit NaNs in recorded periods).
        3. Unlisted / Inactive periods (item not yet launched or carried in store).
        4. Invalid sales (negative quantities).

        Args:
            df: Processed long-format DataFrame.
            allow_negative_sales: Whether negative values (e.g. returns) are permitted.
            enforce_continuity: Whether to strictly check temporal continuity.

        Returns:
            DataQualityReport object.
        """
        report = DataQualityReport(dataset_name="M5_Processed")
        report.total_records = len(df)

        # 1. Column validation
        missing_cols = cls.REQUIRED_PROCESSED_COLS - set(df.columns)
        if missing_cols:
            report.issues.append(f"Missing required processed columns: {sorted(missing_cols)}")
            report.is_valid = False

        if report.total_records == 0:
            report.issues.append("Dataset is empty.")
            report.is_valid = False
            return report

        # 2. Date validation
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                date_series = pd.to_datetime(df["date"])
            except Exception as e:
                report.issues.append(f"Failed to parse 'date' column as datetime: {e}")
                report.is_valid = False
                return report
        else:
            date_series = df["date"]

        report.start_date = str(date_series.min().date())
        report.end_date = str(date_series.max().date())
        expected_unique_dates = pd.date_range(date_series.min(), date_series.max(), freq="D")
        report.total_days = len(expected_unique_dates)

        # 3. Cardinality
        report.unique_items = int(df["item_id"].nunique())
        report.unique_stores = int(df["store_id"].nunique())

        if "id" in df.columns:
            series_col = df["id"]
        else:
            series_col = df["item_id"].astype(str) + "_" + df["store_id"].astype(str)
        report.unique_series = int(series_col.nunique())

        # 4. Duplicate Check on Primary Temporal Key (series_id, date)
        dup_mask = df.duplicated(subset=["item_id", "store_id", "date"], keep=False)
        report.duplicate_records_count = int(dup_mask.sum())
        if report.duplicate_records_count > 0:
            report.issues.append(
                f"Found {report.duplicate_records_count} duplicate records for key (item_id, store_id, date)"
            )
            report.is_valid = False

        # 5. Sales integrity: Zero vs Missing vs Negative
        sales = df["sales"]
        missing_sales = sales.isna()
        report.missing_sales_count = int(missing_sales.sum())
        report.missing_sales_pct = (
            (report.missing_sales_count / report.total_records) * 100 if report.total_records else 0
        )

        valid_sales = sales.dropna()
        neg_sales = (valid_sales < 0).sum()
        report.negative_sales_count = int(neg_sales)
        if not allow_negative_sales and report.negative_sales_count > 0:
            report.issues.append(
                f"Found {report.negative_sales_count} negative sales values."
            )
            report.is_valid = False

        zero_sales = (valid_sales == 0).sum()
        report.zero_sales_count = int(zero_sales)
        report.zero_sales_pct = (
            (report.zero_sales_count / len(valid_sales)) * 100 if len(valid_sales) > 0 else 0
        )

        # 6. Price validation & Unlisted status
        if "sell_price" in df.columns:
            prices = df["sell_price"]
            missing_prices = prices.isna().sum()
            report.missing_price_count = int(missing_prices)
            report.missing_price_pct = (
                (report.missing_price_count / report.total_records) * 100 if report.total_records else 0
            )

            # Inactive/unlisted days: if flagged explicitly or NaN price
            if "is_active" in df.columns:
                report.unlisted_item_days_count = int((~df["is_active"]).sum())
            else:
                report.unlisted_item_days_count = int(missing_prices)

        # 7. Date Continuity Check per Series
        if enforce_continuity:
            # Sample up to 10 series to verify temporal continuity efficiently
            sample_series = df[["item_id", "store_id"]].drop_duplicates().head(10)
            for _, row in sample_series.iterrows():
                sub = df[(df["item_id"] == row["item_id"]) & (df["store_id"] == row["store_id"])]
                actual_dates = pd.to_datetime(sub["date"]).sort_values()
                if len(actual_dates) > 1:
                    date_diffs = actual_dates.diff().dropna()
                    one_day = pd.Timedelta(days=1)
                    if not (date_diffs == one_day).all():
                        report.date_continuity_violations += 1
                        report.issues.append(
                            f"Temporal discontinuity detected for item={row['item_id']}, store={row['store_id']}"
                        )
                        report.is_valid = False
                        break

        # 8. Hierarchical Consistency Check
        # Rule: Each item belongs to exactly 1 dept and 1 cat; each store belongs to 1 state
        item_dept_map = df.groupby("item_id", observed=True)["dept_id"].nunique()
        if (item_dept_map > 1).any():
            report.hierarchy_violations += int((item_dept_map > 1).sum())
            report.issues.append("Items map to multiple departments.")
            report.is_valid = False

        item_cat_map = df.groupby("item_id", observed=True)["cat_id"].nunique()
        if (item_cat_map > 1).any():
            report.hierarchy_violations += int((item_cat_map > 1).sum())
            report.issues.append("Items map to multiple categories.")
            report.is_valid = False

        store_state_map = df.groupby("store_id", observed=True)["state_id"].nunique()
        if (store_state_map > 1).any():
            report.hierarchy_violations += int((store_state_map > 1).sum())
            report.issues.append("Stores map to multiple states.")
            report.is_valid = False

        return report
