"""M5 Dataset Loader and Sample Generator for MarketPulse."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.DataLoader")


class M5DataLoader:
    """Manages discovery, memory-efficient loading, and sample generation for M5 data."""

    EXPECTED_RAW_FILES = {
        "sales": "sales_train_validation.csv",
        "calendar": "calendar.csv",
        "prices": "sell_prices.csv",
    }

    def __init__(self, raw_dir: Union[str, Path]):
        """Initializes loader with path to raw data directory.

        Args:
            raw_dir: Path to raw directory containing M5 CSV files.
        """
        self.raw_dir = Path(raw_dir)

    def check_raw_files_exist(self) -> Dict[str, bool]:
        """Checks whether expected M5 raw files exist.

        Returns:
            Dictionary mapping dataset key to existence boolean.
        """
        status = {}
        for key, filename in self.EXPECTED_RAW_FILES.items():
            status[key] = (self.raw_dir / filename).is_file()
        return status

    def get_missing_files(self) -> List[str]:
        """Returns list of missing raw filenames."""
        status = self.check_raw_files_exist()
        return [self.EXPECTED_RAW_FILES[k] for k, exists in status.items() if not exists]

    def load_calendar(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads calendar.csv with optimized dtypes.

        Args:
            nrows: Optional row limit for fast inspection.

        Returns:
            Calendar DataFrame.
        """
        path = self.raw_dir / self.EXPECTED_RAW_FILES["calendar"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Calendar file missing: {path}. "
                "Download it from Kaggle M5 competition: https://www.kaggle.com/c/m5-forecasting-accuracy/data"
            )

        dtypes = {
            "date": "string",
            "wm_yr_wk": "int16",
            "weekday": "category",
            "wday": "int8",
            "month": "int8",
            "year": "int16",
            "d": "string",
            "event_name_1": "category",
            "event_type_1": "category",
            "event_name_2": "category",
            "event_type_2": "category",
            "snap_CA": "int8",
            "snap_TX": "int8",
            "snap_WI": "int8",
        }
        logger.info(f"Loading calendar from {path}")
        df = pd.read_csv(path, dtype=dtypes, nrows=nrows)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        return df

    def load_prices(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads sell_prices.csv with optimized dtypes.

        Args:
            nrows: Optional row limit.

        Returns:
            Prices DataFrame.
        """
        path = self.raw_dir / self.EXPECTED_RAW_FILES["prices"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Prices file missing: {path}. "
                "Download it from Kaggle M5 competition: https://www.kaggle.com/c/m5-forecasting-accuracy/data"
            )

        dtypes = {
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": "int16",
            "sell_price": "float32",
        }
        logger.info(f"Loading sell prices from {path}")
        return pd.read_csv(path, dtype=dtypes, nrows=nrows)

    def load_sales_raw(
        self,
        nrows: Optional[int] = None,
        store_ids: Optional[List[str]] = None,
        item_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Loads wide sales_train_validation.csv.

        Args:
            nrows: Optional row limit.
            store_ids: Optional list of stores to filter.
            item_ids: Optional list of items to filter.

        Returns:
            Wide sales DataFrame.
        """
        path = self.raw_dir / self.EXPECTED_RAW_FILES["sales"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Sales file missing: {path}. "
                "Download it from Kaggle M5 competition: https://www.kaggle.com/c/m5-forecasting-accuracy/data"
            )

        logger.info(f"Loading raw sales from {path}")
        df = pd.read_csv(path, nrows=nrows)
        if store_ids:
            df = df[df["store_id"].isin(store_ids)]
        if item_ids:
            df = df[df["item_id"].isin(item_ids)]
        return df

    @staticmethod
    def generate_synthetic_sample(
        output_dir: Union[str, Path],
        n_items: int = 10,
        n_stores: int = 2,
        n_days: int = 120,
        random_seed: int = 42,
    ) -> Tuple[Path, Path, Path]:
        """Generates a synthetic M5-compatible dataset for development and automated testing.

        Strictly follows M5 schema, column naming, and hierarchical conventions.

        Args:
            output_dir: Target directory to save CSVs.
            n_items: Number of SKUs to generate.
            n_stores: Number of stores to generate.
            n_days: Number of days to simulate (starting from 2011-01-29).
            random_seed: Seed for reproducibility.

        Returns:
            Tuple of Paths: (sales_path, calendar_path, prices_path)
        """
        rng = np.random.default_rng(random_seed)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        states = ["CA", "TX", "WI"][: max(1, min(3, n_stores))]
        stores = [f"{states[i % len(states)]}_{i // len(states) + 1}" for i in range(n_stores)]
        categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]

        items = []
        for i in range(n_items):
            cat = categories[i % len(categories)]
            dept = f"{cat}_{(i % 3) + 1}"
            item = f"{dept}_{i:03d}"
            items.append((item, dept, cat))

        # 1. Calendar
        start_date = pd.Timestamp("2011-01-29")
        dates = pd.date_range(start=start_date, periods=n_days, freq="D")
        
        # M5 recurring events mapping (month, day) -> (name, type)
        known_events = {
            (2, 6): ("SuperBowl", "Sporting"),
            (2, 14): ("ValentinesDay", "Cultural"),
            (3, 17): ("StPatricksDay", "Cultural"),
            (4, 24): ("Easter", "Cultural"),
            (5, 30): ("MemorialDay", "National"),
            (7, 4): ("IndependenceDay", "National"),
            (9, 5): ("LaborDay", "National"),
            (10, 31): ("Halloween", "Cultural"),
            (11, 24): ("Thanksgiving", "National"),
            (12, 25): ("Christmas", "National"),
            (1, 1): ("NewYear", "National"),
        }

        cal_records = []
        for idx, dt in enumerate(dates):
            d_col = f"d_{idx + 1}"
            # M5 wm_yr_wk convention: 11101 = year 2011, week 01
            wm_yr_wk = (dt.year % 100) * 1000 + 100 + dt.isocalendar()[1]
            event_info = known_events.get((dt.month, dt.day), (None, None))
            cal_records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "wm_yr_wk": wm_yr_wk,
                "weekday": dt.strftime("%A"),
                "wday": (dt.dayofweek + 2) % 7 or 7,
                "month": dt.month,
                "year": dt.year,
                "d": d_col,
                "event_name_1": event_info[0],
                "event_type_1": event_info[1],
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": 1 if dt.day <= 10 else 0,
                "snap_TX": 1 if 5 <= dt.day <= 15 else 0,
                "snap_WI": 1 if 7 <= dt.day <= 17 else 0,
            })
        calendar_df = pd.DataFrame(cal_records)
        calendar_path = out / "calendar.csv"
        calendar_df.to_csv(calendar_path, index=False)

        # 2. Sales Wide Matrix with Diverse Demand Regimes
        # Assign items to archetypes: Smooth, Intermittent, Erratic, Lumpy
        sales_records = []
        d_cols = [f"d_{i+1}" for i in range(n_days)]
        
        for s_idx, store in enumerate(stores):
            state = store.split("_")[0]
            for item_idx, (item, dept, cat) in enumerate(items):
                series_id = f"{item}_{store}_validation"
                regime_type = item_idx % 4  # 0: Smooth, 1: Intermittent, 2: Erratic, 3: Lumpy
                
                # Base parameters
                if regime_type == 0:  # Smooth: steady demand, low zero count
                    lam = 5.0 + rng.uniform(1.0, 3.0)
                    zero_prob = 0.05
                    var_factor = 1.0
                elif regime_type == 1:  # Intermittent: infrequent demand, consistent order size
                    lam = 1.5
                    zero_prob = 0.55
                    var_factor = 0.8
                elif regime_type == 2:  # Erratic: frequent demand, high variance bursts
                    lam = 4.0
                    zero_prob = 0.10
                    var_factor = 3.5
                else:  # Lumpy: infrequent demand, high variance bursts
                    lam = 2.0
                    zero_prob = 0.65
                    var_factor = 4.0
                
                base_sales = np.zeros(n_days, dtype=int)
                for d_i, dt in enumerate(dates):
                    # Weekend lift (+30% on Sat/Sun)
                    w_lift = 1.35 if dt.dayofweek in [5, 6] else 1.0
                    # Event lift (+50% if holiday)
                    ev_lift = 1.50 if (dt.month, dt.day) in known_events else 1.0
                    
                    eff_lam = lam * w_lift * ev_lift
                    if rng.random() < zero_prob:
                        base_sales[d_i] = 0
                    else:
                        if var_factor > 1.0 and rng.random() < 0.25:
                            # Volatile burst
                            base_sales[d_i] = int(rng.poisson(lam=eff_lam * var_factor))
                        else:
                            base_sales[d_i] = max(1, int(rng.poisson(lam=eff_lam)))

                rec = {
                    "id": series_id,
                    "item_id": item,
                    "dept_id": dept,
                    "cat_id": cat,
                    "store_id": store,
                    "state_id": state,
                }
                for d_c, val in zip(d_cols, base_sales):
                    rec[d_c] = int(val)
                sales_records.append(rec)

        sales_df = pd.DataFrame(sales_records)
        sales_path = out / "sales_train_validation.csv"
        sales_df.to_csv(sales_path, index=False)

        # 3. Sell Prices with Promotional Variation
        weeks = calendar_df["wm_yr_wk"].unique()
        price_records = []
        for store in stores:
            for item, _, _ in items:
                base_price = round(float(rng.uniform(2.5, 18.0)), 2)
                for w in weeks:
                    # 15% probability of a promotional discount
                    promo = 0.80 if rng.random() < 0.15 else 1.0
                    price = round(base_price * promo, 2)
                    price_records.append({
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": w,
                        "sell_price": price,
                    })

        prices_df = pd.DataFrame(price_records)
        prices_path = out / "sell_prices.csv"
        prices_df.to_csv(prices_path, index=False)

        logger.info(
            f"Generated synthetic M5 dataset at {out} with "
            f"{len(sales_df)} series, {n_days} days, {len(prices_df)} price records."
        )
        return sales_path, calendar_path, prices_path

