"""Hierarchical structure builder, validation, and summing matrix construction for M5."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.hierarchy import M5Hierarchy
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Hierarchy.Builder")


class HierarchyTree:
    """Represents the M5 retail hierarchy tree: State -> Store -> Category -> Department -> Item."""

    def __init__(self, metadata_df: pd.DataFrame):
        """Initializes tree from dataframe containing hierarchy metadata.

        Args:
            metadata_df: DataFrame containing hierarchy metadata columns.
        """
        self.m5_hierarchy = M5Hierarchy(metadata_df=metadata_df)
        self.metadata = self.m5_hierarchy.metadata
        self.tree_status = self.m5_hierarchy.validate_strict_tree_property()
        if not all(self.tree_status.values()):
            raise ValueError(f"Hierarchy violation detected: {self.tree_status}")

    def build_summing_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Constructs the dense summing matrix S such that y_total = S * y_bottom.

        Returns:
            Tuple of:
            - S: 2D numpy array of shape (m_total, n_bottom)
            - total_node_names: List of all node identifiers across levels
            - bottom_node_names: List of bottom-level series identifiers
        """
        s_sparse, total_names, bottom_names = self.m5_hierarchy.build_summing_matrix()
        return s_sparse.toarray(), total_names, bottom_names

    def get_parent_child_mapping(self) -> Dict[str, List[str]]:
        """Maps each parent node to its immediate child nodes in the hierarchy tree.

        Returns:
            Dictionary mapping parent node identifier to list of child node identifiers.
        """
        mapping: Dict[str, List[str]] = {}

        # 1. Root 'Total' -> States
        states = sorted(self.metadata["state_id"].unique().tolist())
        mapping["Total"] = [str(s) for s in states]

        # 2. State -> Stores
        for state in states:
            stores = sorted(self.metadata[self.metadata["state_id"] == state]["store_id"].unique().tolist())
            mapping[str(state)] = [str(st) for st in stores]

        # 3. Store -> Store_Dept
        for store in sorted(self.metadata["store_id"].unique()):
            store_df = self.metadata[self.metadata["store_id"] == store]
            depts = sorted(store_df["dept_id"].unique().tolist())
            mapping[str(store)] = [f"{store}_{d}" for d in depts]

        # 4. Store_Dept -> Store_Item (bottom)
        for store in sorted(self.metadata["store_id"].unique()):
            store_df = self.metadata[self.metadata["store_id"] == store]
            for dept in sorted(store_df["dept_id"].unique()):
                dept_df = store_df[store_df["dept_id"] == dept]
                items = sorted(dept_df["item_id"].unique().tolist())
                parent_key = f"{store}_{dept}"
                mapping[parent_key] = [f"{store}_{item}" for item in items]

        # Product hierarchy branch: Total -> Categories -> Departments
        cats = sorted(self.metadata["cat_id"].unique().tolist())
        mapping["Total_Categories"] = [str(c) for c in cats]
        for cat in cats:
            depts = sorted(self.metadata[self.metadata["cat_id"] == cat]["dept_id"].unique().tolist())
            mapping[str(cat)] = [str(d) for d in depts]

        return mapping

    def get_level_summary(self) -> pd.DataFrame:
        """Returns node count and grouping details for all 12 hierarchy levels."""
        records = []
        for lvl in range(1, 13):
            defn = self.m5_hierarchy.LEVEL_DEFINITIONS[lvl]
            nodes = self.m5_hierarchy.get_level_nodes(lvl)
            records.append({
                "level": lvl,
                "name": defn["name"],
                "group_cols": ", ".join(defn["group_cols"]) if defn["group_cols"] else "None (Root)",
                "node_count": len(nodes),
            })
        return pd.DataFrame(records)
