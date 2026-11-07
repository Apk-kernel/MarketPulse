"""Hierarchical structure definitions, validation, and roll-up mappings for M5."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Hierarchy")


class M5Hierarchy:
    """Manages the M5 multi-level retail hierarchy and aggregation relationships."""

    # Official 12 M5 Evaluation Levels
    LEVEL_DEFINITIONS: Dict[int, Dict[str, object]] = {
        1: {"name": "Total", "group_cols": []},
        2: {"name": "State", "group_cols": ["state_id"]},
        3: {"name": "Store", "group_cols": ["store_id"]},
        4: {"name": "Category", "group_cols": ["cat_id"]},
        5: {"name": "Department", "group_cols": ["dept_id"]},
        6: {"name": "State_Category", "group_cols": ["state_id", "cat_id"]},
        7: {"name": "State_Department", "group_cols": ["state_id", "dept_id"]},
        8: {"name": "Store_Category", "group_cols": ["store_id", "cat_id"]},
        9: {"name": "Store_Department", "group_cols": ["store_id", "dept_id"]},
        10: {"name": "Item", "group_cols": ["item_id"]},
        11: {"name": "State_Item", "group_cols": ["state_id", "item_id"]},
        12: {"name": "Bottom (Store_Item)", "group_cols": ["store_id", "item_id"]},
    }

    HIERARCHICAL_PATH = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]

    def __init__(self, metadata_df: pd.DataFrame):
        """Initializes hierarchy from series metadata.

        Args:
            metadata_df: DataFrame containing at least ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'].
        """
        self._validate_metadata_columns(metadata_df)
        self.metadata = (
            metadata_df[self.HIERARCHICAL_PATH].drop_duplicates().reset_index(drop=True)
        )
        self.n_bottom_series = len(self.metadata)
        logger.info(f"Initialized M5Hierarchy with {self.n_bottom_series:,} bottom-level series.")

    @classmethod
    def _validate_metadata_columns(cls, df: pd.DataFrame) -> None:
        missing = set(cls.HIERARCHICAL_PATH) - set(df.columns)
        if missing:
            raise ValueError(f"Metadata DataFrame is missing hierarchy columns: {sorted(missing)}")

    def validate_strict_tree_property(self) -> Dict[str, bool]:
        """Validates that each lower node maps to exactly one parent node in the hierarchy.

        Returns:
            Dictionary of validation checks:
            - item_to_dept: True if each item belongs to exactly one department
            - dept_to_cat: True if each department belongs to exactly one category
            - store_to_state: True if each store belongs to exactly one state
        """
        results = {}
        # 1. Item -> Department
        item_depts = self.metadata.groupby("item_id", observed=True)["dept_id"].nunique()
        results["item_to_dept"] = bool((item_depts <= 1).all())

        # 2. Dept -> Category
        dept_cats = self.metadata.groupby("dept_id", observed=True)["cat_id"].nunique()
        results["dept_to_cat"] = bool((dept_cats <= 1).all())

        # 3. Store -> State
        store_states = self.metadata.groupby("store_id", observed=True)["state_id"].nunique()
        results["store_to_state"] = bool((store_states <= 1).all())

        is_valid = all(results.values())
        if not is_valid:
            logger.warning(f"Hierarchy violation detected: {results}")
        else:
            logger.info("Hierarchy verified: Strict tree properties satisfied.")
        return results

    def get_level_nodes(self, level: int) -> List[str]:
        """Returns unique node identifiers for a specified M5 hierarchy level.

        Args:
            level: Level integer (1 to 12).

        Returns:
            List of string identifiers.
        """
        if level not in self.LEVEL_DEFINITIONS:
            raise ValueError(f"Invalid level: {level}. Expected 1 to 12.")

        group_cols = self.LEVEL_DEFINITIONS[level]["group_cols"]
        if not group_cols:
            return ["Total"]

        grouped = self.metadata[group_cols].drop_duplicates()
        if len(group_cols) == 1:
            return grouped[group_cols[0]].astype(str).tolist()
        else:
            return grouped.apply(lambda row: "_".join(row.values.astype(str)), axis=1).tolist()

    def aggregate_bottom_up(
        self,
        bottom_df: pd.DataFrame,
        level: int,
        date_col: str = "date",
        sales_col: str = "sales",
    ) -> pd.DataFrame:
        """Aggregates bottom-level demand to a specified hierarchy level.

        Args:
            bottom_df: Long-format bottom-level demand DataFrame.
            level: Target level (1 to 12).
            date_col: Name of date column.
            sales_col: Name of sales column.

        Returns:
            Aggregated DataFrame with [date, <group_cols>, sales].
        """
        if level not in self.LEVEL_DEFINITIONS:
            raise ValueError(f"Invalid level: {level}. Expected 1 to 12.")

        group_cols = self.LEVEL_DEFINITIONS[level]["group_cols"]
        if not group_cols:
            # Level 1: Total
            aggregated = (
                bottom_df.groupby(date_col, observed=True)[sales_col]
                .sum()
                .reset_index()
            )
            aggregated["series_id"] = "Total"
            return aggregated

        agg_keys = [date_col] + group_cols
        aggregated = (
            bottom_df.groupby(agg_keys, observed=True)[sales_col]
            .sum()
            .reset_index()
        )
        if len(group_cols) == 1:
            aggregated["series_id"] = aggregated[group_cols[0]].astype(str)
        else:
            aggregated["series_id"] = aggregated[group_cols].apply(
                lambda row: "_".join(row.values.astype(str)), axis=1
            )
        return aggregated

    def build_summing_matrix(self) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
        """Constructs the hierarchical summing matrix S such that y_all = S * y_bottom.

        Returns:
            Tuple of:
            - S: Sparse CSR matrix of shape (n_total_series, n_bottom_series)
            - total_series_names: List of all series names across levels 1 through 12
            - bottom_series_names: List of bottom-level series names
        """
        # Define bottom series order
        bottom_df = self.metadata.copy()
        bottom_df["bottom_id"] = bottom_df["store_id"].astype(str) + "_" + bottom_df["item_id"].astype(str)
        bottom_series_names = bottom_df["bottom_id"].tolist()
        n_bottom = len(bottom_series_names)
        bottom_to_idx = {name: i for i, name in enumerate(bottom_series_names)}

        total_series_names = []
        rows = []
        cols = []
        current_row = 0

        for level in range(1, 13):
            group_cols = self.LEVEL_DEFINITIONS[level]["group_cols"]
            if not group_cols:
                # Level 1: Total (sums all bottom series)
                total_series_names.append("Total")
                for c in range(n_bottom):
                    rows.append(current_row)
                    cols.append(c)
                current_row += 1
            else:
                # Group bottom series
                grouped = bottom_df.groupby(group_cols, observed=True)
                for group_key, indices in grouped.groups.items():
                    if isinstance(group_key, tuple):
                        node_name = "_".join(str(k) for k in group_key)
                    else:
                        node_name = str(group_key)
                    total_series_names.append(node_name)
                    for idx in indices:
                        rows.append(current_row)
                        cols.append(idx)
                    current_row += 1

        data = np.ones(len(rows), dtype=np.float32)
        s_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(current_row, n_bottom))
        logger.info(
            f"Built summing matrix S with shape {s_matrix.shape} "
            f"({current_row} total aggregated nodes from {n_bottom} bottom nodes)."
        )
        return s_matrix, total_series_names, bottom_series_names

