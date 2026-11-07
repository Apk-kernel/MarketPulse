"""Hierarchical aggregation, top-down disaggregation, and forecast coherence verification."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.hierarchy import M5Hierarchy
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Hierarchy.Aggregation")


class HierarchicalAggregator:
    """Handles bottom-up aggregation, top-down disaggregation, and coherence checking."""

    def __init__(self, hierarchy: M5Hierarchy):
        """Initializes aggregator with an M5Hierarchy instance.

        Args:
            hierarchy: Configured M5Hierarchy containing metadata and levels.
        """
        self.hierarchy = hierarchy
        self.metadata = hierarchy.metadata
        self.s_matrix, self.total_node_names, self.bottom_node_names = (
            hierarchy.build_summing_matrix()
        )
        self.n_total = len(self.total_node_names)
        self.n_bottom = len(self.bottom_node_names)
        self._s_dense = self.s_matrix.toarray()

    def aggregate_bottom_to_all_levels(
        self,
        bottom_wide_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregates bottom-level demand to all hierarchy levels using summing matrix S.

        Args:
            bottom_wide_df: DataFrame indexed by date with columns matching bottom_node_names.

        Returns:
            DataFrame indexed by date with columns matching total_node_names.
        """
        missing = set(self.bottom_node_names) - set(bottom_wide_df.columns)
        if missing:
            raise ValueError(f"bottom_wide_df missing {len(missing)} required bottom nodes.")

        # Ensure correct column ordering
        y_bottom = bottom_wide_df[self.bottom_node_names].values  # Shape: (T, n_bottom)

        # y_all = y_bottom * S^T -> Shape: (T, n_total)
        y_all = y_bottom @ self._s_dense.T

        return pd.DataFrame(
            y_all,
            index=bottom_wide_df.index,
            columns=self.total_node_names,
        )

    def prepare_bottom_wide_dataframe(
        self,
        long_df: pd.DataFrame,
        date_col: str = "date",
        sales_col: str = "sales",
    ) -> pd.DataFrame:
        """Converts long M5 DataFrame into a wide date-indexed DataFrame of bottom-level series.

        Bottom series identifier is formed as: f"{store_id}_{item_id}".

        Args:
            long_df: Long DataFrame containing [date_col, store_id, item_id, sales_col].
            date_col: Column name for date.
            sales_col: Column name for sales.

        Returns:
            Wide DataFrame indexed by date with bottom series as columns.
        """
        df = long_df.copy()
        df["bottom_id"] = df["store_id"].astype(str) + "_" + df["item_id"].astype(str)
        pivoted = df.pivot(index=date_col, columns="bottom_id", values=sales_col).fillna(0.0)
        pivoted.index = pd.to_datetime(pivoted.index)
        if len(pivoted) > 1:
            # Ensure frequency is set for sktime estimators
            try:
                pivoted.index = pd.date_range(start=pivoted.index[0], periods=len(pivoted), freq="D")
            except Exception:
                pass

        # Align with canonical bottom node ordering
        for col in self.bottom_node_names:
            if col not in pivoted.columns:
                pivoted[col] = 0.0

        return pivoted[self.bottom_node_names]

    def compute_historical_proportions(
        self,
        bottom_wide_df: pd.DataFrame,
        method: str = "pha",
    ) -> np.ndarray:
        """Computes bottom-level historical volume proportions for top-down disaggregation.

        Args:
            bottom_wide_df: Wide DataFrame of bottom-level historical sales.
            method: Proportion method:
                - 'pha': Proportion of Historical Averages (sum(y_i) / sum(y_total))
                - 'ahp': Average Historical Proportions (mean_t(y_{i,t} / y_{total,t}))

        Returns:
            1D numpy array of shape (n_bottom,) summing to 1.0.
        """
        y_bottom = bottom_wide_df[self.bottom_node_names].values  # Shape: (T, n_bottom)
        y_total = y_bottom.sum(axis=1)  # Shape: (T,)

        if method == "pha":
            total_sum = y_total.sum()
            if total_sum <= 0:
                return np.ones(self.n_bottom, dtype=np.float64) / self.n_bottom
            proportions = y_bottom.sum(axis=0) / total_sum

        elif method == "ahp":
            valid_mask = y_total > 0
            if not np.any(valid_mask):
                return np.ones(self.n_bottom, dtype=np.float64) / self.n_bottom
            daily_props = y_bottom[valid_mask] / y_total[valid_mask, np.newaxis]
            proportions = daily_props.mean(axis=0)

        else:
            raise ValueError(f"Unknown proportion method '{method}'. Choose 'pha' or 'ahp'.")

        # Normalize to ensure exact sum to 1.0
        prop_sum = proportions.sum()
        if prop_sum > 0:
            proportions /= prop_sum
        else:
            proportions = np.ones(self.n_bottom, dtype=np.float64) / self.n_bottom

        return proportions

    def disaggregate_top_down(
        self,
        top_forecasts: np.ndarray,
        proportions: np.ndarray,
    ) -> np.ndarray:
        """Disaggregates top-level forecast down to bottom-level series using proportions.

        Args:
            top_forecasts: 1D array of shape (H,) or 2D array of shape (H, 1) of top-level demand.
            proportions: 1D array of shape (n_bottom,) summing to 1.0.

        Returns:
            2D numpy array of shape (H, n_bottom).
        """
        top_1d = np.asarray(top_forecasts).reshape(-1, 1)
        props_1d = np.asarray(proportions).reshape(1, -1)
        return top_1d @ props_1d

    def verify_coherence(
        self,
        forecasts_wide_df: pd.DataFrame,
        tolerance: float = 1e-4,
    ) -> Dict[str, Any]:
        """Mathematically verifies hierarchical forecast coherence across all levels.

        For coherent forecasts, the vector of all predictions y_hat satisfies:
            y_hat = S * y_hat_bottom.
        Any deviation e = y_hat - S * y_hat_bottom represents hierarchical incoherence.

        Args:
            forecasts_wide_df: DataFrame indexed by forecast horizon/date with columns matching total_node_names.
            tolerance: Maximum absolute numerical discrepancy threshold for coherence.

        Returns:
            Dictionary containing:
                - is_coherent: bool, True if max absolute error <= tolerance
                - max_abs_incoherence: float, maximum absolute deviation
                - mean_abs_incoherence: float, average absolute deviation across all nodes and steps
                - relative_incoherence: float, Frobenius norm of error divided by norm of forecasts
                - level_incoherences: Dict[int, float], max absolute incoherence per M5 hierarchy level
        """
        missing = set(self.total_node_names) - set(forecasts_wide_df.columns)
        if missing:
            raise ValueError(f"forecasts_wide_df missing {len(missing)} nodes from total_node_names.")

        y_all = forecasts_wide_df[self.total_node_names].values  # Shape: (H, n_total)
        y_bottom = forecasts_wide_df[self.bottom_node_names].values  # Shape: (H, n_bottom)

        # Expected coherent forecasts from bottom-up summing: y_bottom @ S.T -> (H, n_total)
        y_expected = y_bottom @ self._s_dense.T

        # Incoherence matrix: (H, n_total)
        diff = np.abs(y_all - y_expected)

        max_err = float(np.max(diff))
        mean_err = float(np.mean(diff))
        norm_all = float(np.linalg.norm(y_all))
        rel_err = float(np.linalg.norm(diff) / norm_all) if norm_all > 0 else 0.0

        # Compute per-level incoherence
        level_errs: Dict[int, float] = {}
        col_names = self.total_node_names

        for lvl in range(1, 13):
            lvl_nodes = set(self.hierarchy.get_level_nodes(lvl))
            lvl_indices = [i for i, name in enumerate(col_names) if name in lvl_nodes]
            if lvl_indices:
                level_errs[lvl] = float(np.max(diff[:, lvl_indices]))
            else:
                level_errs[lvl] = 0.0

        is_coherent = bool(max_err <= tolerance)

        return {
            "is_coherent": is_coherent,
            "max_abs_incoherence": max_err,
            "mean_abs_incoherence": mean_err,
            "relative_incoherence": rel_err,
            "level_incoherences": level_errs,
        }

    def to_sktime_hierarchical(
        self,
        long_df: pd.DataFrame,
        hierarchy_cols: Optional[List[str]] = None,
        date_col: str = "date",
        sales_col: str = "sales",
    ) -> pd.DataFrame:
        """Converts long M5 DataFrame into a sktime MultiIndex hierarchical DataFrame with Aggregator applied.

        Args:
            long_df: Long DataFrame containing hierarchy columns, date, and sales.
            hierarchy_cols: List of hierarchy level columns in top-to-bottom order.
                Defaults to ['state_id', 'store_id'].
            date_col: Date column name.
            sales_col: Target demand column name.

        Returns:
            MultiIndex DataFrame with ['__total'] levels added by sktime Aggregator.
        """
        from sktime.transformations.hierarchical.aggregate import Aggregator

        if hierarchy_cols is None:
            hierarchy_cols = ["state_id", "store_id"]

        df = long_df[hierarchy_cols + [date_col, sales_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Group duplicate timestamps within the same hierarchy leaf
        grouped = df.groupby(hierarchy_cols + [date_col], observed=True)[sales_col].sum().reset_index()
        multi_indexed = grouped.set_index(hierarchy_cols + [date_col]).sort_index()

        agg = Aggregator()
        hier_df = agg.fit_transform(multi_indexed)
        return hier_df
