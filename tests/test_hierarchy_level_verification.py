"""Mathematical verification and exhaustive level-by-level automated tests for all 12 M5 hierarchy levels."""

import warnings
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from src.data.hierarchy import M5Hierarchy
from src.hierarchy.aggregation import HierarchicalAggregator
from src.hierarchy.hierarchy_builder import HierarchyTree
from src.hierarchy.reconciliation import HierarchicalReconciler


@pytest.fixture(scope="module")
def multi_level_hierarchy_fixture():
    """Generates synthetic M5 data across 2 states, 3 stores, 3 categories, 4 departments, and 6 items."""
    dates = pd.date_range("2016-01-01", periods=35, freq="D")
    rng = np.random.default_rng(777)

    hierarchy_catalog = [
        # state, store, cat, dept, item
        ("CA", "CA_1", "FOODS", "FOODS_1", "FOODS_1_001"),
        ("CA", "CA_1", "FOODS", "FOODS_1", "FOODS_1_002"),
        ("CA", "CA_1", "FOODS", "FOODS_2", "FOODS_2_001"),
        ("CA", "CA_1", "HOBBIES", "HOBBIES_1", "HOBBIES_1_001"),
        ("CA", "CA_2", "FOODS", "FOODS_1", "FOODS_1_001"),
        ("CA", "CA_2", "FOODS", "FOODS_2", "FOODS_2_001"),
        ("CA", "CA_2", "HOUSEHOLD", "HOUSEHOLD_1", "HOUSEHOLD_1_001"),
        ("TX", "TX_1", "FOODS", "FOODS_1", "FOODS_1_001"),
        ("TX", "TX_1", "HOBBIES", "HOBBIES_1", "HOBBIES_1_001"),
        ("TX", "TX_1", "HOUSEHOLD", "HOUSEHOLD_1", "HOUSEHOLD_1_001"),
    ]

    records = []
    for state, store, cat, dept, item in hierarchy_catalog:
        mean_demand = rng.uniform(4.0, 18.0)
        for d in dates:
            sales = max(0.0, float(mean_demand + rng.normal(0, 1.5)))
            records.append({
                "date": d,
                "state_id": state,
                "store_id": store,
                "cat_id": cat,
                "dept_id": dept,
                "item_id": item,
                "sales": sales,
            })

    df = pd.DataFrame(records)
    tree = HierarchyTree(metadata_df=df)
    hierarchy = tree.m5_hierarchy
    aggregator = HierarchicalAggregator(hierarchy=hierarchy)

    bottom_wide = aggregator.prepare_bottom_wide_dataframe(df)
    all_wide = aggregator.aggregate_bottom_to_all_levels(bottom_wide)
    S, total_names, bottom_names = tree.build_summing_matrix()

    return {
        "df": df,
        "tree": tree,
        "hierarchy": hierarchy,
        "aggregator": aggregator,
        "bottom_wide": bottom_wide,
        "all_wide": all_wide,
        "S": S,
        "total_names": total_names,
        "bottom_names": bottom_names,
    }


# ============================================================================
# 1. Automated Level-by-Level Tests for ALL 12 M5 Hierarchy Levels
# ============================================================================

ALL_LEVELS = list(range(1, 13))


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_summing_matrix_level_exact_structure(level, multi_level_hierarchy_fixture):
    """Verifies that Summing Matrix S accurately encodes aggregation sets for Level {level}."""
    hierarchy: M5Hierarchy = multi_level_hierarchy_fixture["hierarchy"]
    S: np.ndarray = multi_level_hierarchy_fixture["S"]
    total_names: List[str] = multi_level_hierarchy_fixture["total_names"]
    bottom_names: List[str] = multi_level_hierarchy_fixture["bottom_names"]
    metadata: pd.DataFrame = hierarchy.metadata

    level_defn = hierarchy.LEVEL_DEFINITIONS[level]
    group_cols = level_defn["group_cols"]
    level_nodes = hierarchy.get_level_nodes(level)

    assert len(level_nodes) > 0, f"Level {level} produced no node names."

    for node_name in level_nodes:
        assert node_name in total_names, f"Node '{node_name}' at Level {level} missing from S rows."
        row_idx = total_names.index(node_name)
        row_vec = S[row_idx, :]

        # Find which bottom series should contribute to this node according to metadata
        if not group_cols:
            expected_leaf_indices = list(range(len(bottom_names)))
        else:
            if len(group_cols) == 1:
                matching_mask = metadata[group_cols[0]].astype(str) == node_name
            else:
                combined_keys = metadata[group_cols].apply(
                    lambda r: "_".join(r.values.astype(str)), axis=1
                )
                matching_mask = combined_keys == node_name

            matching_bottom_ids = set(
                metadata.loc[matching_mask, "store_id"].astype(str)
                + "_"
                + metadata.loc[matching_mask, "item_id"].astype(str)
            )
            expected_leaf_indices = [
                i for i, b_name in enumerate(bottom_names) if b_name in matching_bottom_ids
            ]

        actual_leaf_indices = np.where(row_vec > 0)[0].tolist()

        assert sorted(actual_leaf_indices) == sorted(expected_leaf_indices), (
            f"Level {level} node '{node_name}' row in S does not match canonical leaves! "
            f"Expected {expected_leaf_indices}, got {actual_leaf_indices}."
        )
        assert np.sum(row_vec) == len(expected_leaf_indices), (
            f"Row sum in S for Level {level} node '{node_name}' must equal leaf count."
        )


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_level_aggregation_invariance(level, multi_level_hierarchy_fixture):
    """Verifies that matrix aggregation y_all = y_bottom * S^T matches groupby at Level {level}."""
    hierarchy: M5Hierarchy = multi_level_hierarchy_fixture["hierarchy"]
    df: pd.DataFrame = multi_level_hierarchy_fixture["df"]
    all_wide: pd.DataFrame = multi_level_hierarchy_fixture["all_wide"]

    # Compute aggregation via pandas groupby
    gb_df = hierarchy.aggregate_bottom_up(df, level=level, date_col="date", sales_col="sales")

    for sid in gb_df["series_id"].unique():
        s_expected = gb_df[gb_df["series_id"] == sid].sort_values("date")["sales"].values
        s_actual = all_wide[sid].values
        assert np.allclose(s_actual, s_expected, atol=1e-5), (
            f"Level {level} node '{sid}' aggregated values diverge between S matrix and pandas groupby!"
        )


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_level_reconciliation_coherence(level, multi_level_hierarchy_fixture):
    """Verifies that reconciled forecasts are strictly coherent for all nodes at Level {level}."""
    hierarchy: M5Hierarchy = multi_level_hierarchy_fixture["hierarchy"]
    all_wide: pd.DataFrame = multi_level_hierarchy_fixture["all_wide"]
    total_names: List[str] = multi_level_hierarchy_fixture["total_names"]
    bottom_names: List[str] = multi_level_hierarchy_fixture["bottom_names"]
    S: np.ndarray = multi_level_hierarchy_fixture["S"]

    # Generate incoherent base forecasts
    rng = np.random.RandomState(42)
    incoherent_base = all_wide.iloc[:7].copy() + rng.normal(0, 3.0, (7, len(total_names)))

    reconciler = HierarchicalReconciler(hierarchy=hierarchy, method="mint_shrink", non_negative=False)
    reconciler.fit(in_sample_actuals=all_wide.iloc[:20], in_sample_base_forecasts=all_wide.iloc[:20] + 1.0)
    reconciled = reconciler.reconcile(incoherent_base)

    level_nodes = hierarchy.get_level_nodes(level)
    rec_bottom = reconciled[bottom_names].values  # (H, n)

    for node in level_nodes:
        row_idx = total_names.index(node)
        s_row = S[row_idx, :]  # (n,)
        expected_node_demand = rec_bottom @ s_row  # (H,)
        actual_node_demand = reconciled[node].values  # (H,)

        assert np.allclose(actual_node_demand, expected_node_demand, atol=1e-5), (
            f"Level {level} node '{node}' is incoherent after MinT reconciliation!"
        )


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_level_reconciliation_idempotency(level, multi_level_hierarchy_fixture):
    """Verifies idempotency at Level {level}: coherent input forecasts must be preserved exactly."""
    hierarchy: M5Hierarchy = multi_level_hierarchy_fixture["hierarchy"]
    all_wide: pd.DataFrame = multi_level_hierarchy_fixture["all_wide"]

    # Perfectly coherent input series from historical ground truth
    coherent_input = all_wide.iloc[:5].copy()

    ols_reconciler = HierarchicalReconciler(hierarchy=hierarchy, method="ols", non_negative=False)
    ols_reconciler.fit()
    reconciled_ols = ols_reconciler.reconcile(coherent_input)

    level_nodes = hierarchy.get_level_nodes(level)
    for node in level_nodes:
        orig = coherent_input[node].values
        recon = reconciled_ols[node].values
        assert np.allclose(orig, recon, atol=1e-5), (
            f"Idempotency violation at Level {level} for node '{node}'!"
        )


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_level_non_negativity_preservation(level, multi_level_hierarchy_fixture):
    """Verifies that NN-MinT preserves non-negativity across all series at Level {level}."""
    hierarchy: M5Hierarchy = multi_level_hierarchy_fixture["hierarchy"]
    all_wide: pd.DataFrame = multi_level_hierarchy_fixture["all_wide"]

    # Severe negative inputs
    negative_base = all_wide.iloc[:5].copy() - 100.0

    nn_reconciler = HierarchicalReconciler(hierarchy=hierarchy, method="mint_shrink", non_negative=True)
    nn_reconciler.fit(in_sample_actuals=all_wide.iloc[:20], in_sample_base_forecasts=all_wide.iloc[:20])
    reconciled_nn = nn_reconciler.reconcile(negative_base)

    level_nodes = hierarchy.get_level_nodes(level)
    for node in level_nodes:
        vals = reconciled_nn[node].values
        assert (vals >= -1e-6).all(), (
            f"Negative forecast detected at Level {level} for node '{node}'!"
        )


# ============================================================================
# 2. Fundamental Algebraic Verification of Linear Reconciliation
# ============================================================================

def test_algebraic_idempotency_s_p_s_equals_s(multi_level_hierarchy_fixture):
    """Algebraic proof verification: S * P * S == S for unbiased linear reconcilers."""
    hierarchy: M5Hierarchy = multi_level_hierarchy_fixture["hierarchy"]
    all_wide: pd.DataFrame = multi_level_hierarchy_fixture["all_wide"]
    S: np.ndarray = multi_level_hierarchy_fixture["S"]

    # 1. OLS
    ols_rec = HierarchicalReconciler(hierarchy=hierarchy, method="ols")
    ols_rec.fit()
    assert HierarchicalReconciler.verify_reconciler_idempotency(S, ols_rec._p_matrix)

    # 2. WLS Structural
    wls_rec = HierarchicalReconciler(hierarchy=hierarchy, method="wls_str")
    wls_rec.fit()
    assert HierarchicalReconciler.verify_reconciler_idempotency(S, wls_rec._p_matrix)

    # 3. Bottom-Up
    bu_rec = HierarchicalReconciler(hierarchy=hierarchy, method="bu")
    bu_rec.fit()
    assert HierarchicalReconciler.verify_reconciler_idempotency(S, bu_rec._p_matrix)

    # 4. MinT Shrinkage
    mint_rec = HierarchicalReconciler(hierarchy=hierarchy, method="mint_shrink")
    mint_rec.fit(in_sample_actuals=all_wide, in_sample_base_forecasts=all_wide + 1.0)
    assert HierarchicalReconciler.verify_reconciler_idempotency(S, mint_rec._p_matrix)


def test_covariance_matrix_properties_and_shrinkage(multi_level_hierarchy_fixture):
    """Verifies that MinT covariance shrinkage matrix is strictly symmetric and positive definite."""
    hierarchy: M5Hierarchy = multi_level_hierarchy_fixture["hierarchy"]
    all_wide: pd.DataFrame = multi_level_hierarchy_fixture["all_wide"]

    mint_rec = HierarchicalReconciler(hierarchy=hierarchy, method="mint_shrink")
    mint_rec.fit(in_sample_actuals=all_wide, in_sample_base_forecasts=all_wide + 1.0)

    W = mint_rec._w_matrix
    cov_diag = HierarchicalReconciler.verify_covariance_matrix(W)

    assert cov_diag["is_symmetric"] is True, "Covariance matrix W must be symmetric!"
    assert cov_diag["is_positive_definite"] is True, "Covariance matrix W must be positive definite!"
    assert cov_diag["min_eigenvalue"] > 0, "All eigenvalues of W must be strictly positive!"
    assert np.isfinite(cov_diag["condition_number"]), "Condition number must be finite!"

    # Verify shrinkage intensity lambda is bounded in [0, 1]
    assert mint_rec.shrinkage_intensity_ is not None
    assert 0.0 <= mint_rec.shrinkage_intensity_ <= 1.0

