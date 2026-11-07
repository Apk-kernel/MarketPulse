"""Unit and integration tests for hierarchical tree, aggregation, and forecast reconciliation."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from src.data.hierarchy import M5Hierarchy
from src.data.loader import M5DataLoader
from src.hierarchy.aggregation import HierarchicalAggregator
from src.hierarchy.hierarchy_builder import HierarchyTree
from src.hierarchy.reconciliation import HierarchicalReconciler


@pytest.fixture(scope="module")
def sample_hierarchy():
    """Generates a reproducible synthetic M5 dataset and hierarchy for tests."""
    dates = pd.date_range("2016-01-01", periods=45, freq="D")
    records = []
    rng = np.random.default_rng(42)

    states_stores = [("CA", "CA_1"), ("CA", "CA_2"), ("TX", "TX_1")]
    items_depts = [
        ("FOODS_1_001", "FOODS_1", "FOODS"),
        ("FOODS_1_002", "FOODS_1", "FOODS"),
        ("FOODS_2_001", "FOODS_2", "FOODS"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES"),
        ("HOUSEHOLD_1_001", "HOUSEHOLD_1", "HOUSEHOLD"),
    ]

    for state, store in states_stores:
        for item, dept, cat in items_depts:
            base_sales = rng.uniform(5.0, 15.0)
            for d in dates:
                sales = max(0.0, base_sales + rng.normal(0, 2.0))
                records.append({
                    "date": d,
                    "state_id": state,
                    "store_id": store,
                    "cat_id": cat,
                    "dept_id": dept,
                    "item_id": item,
                    "sales": float(sales),
                })

    df = pd.DataFrame(records)
    tree = HierarchyTree(metadata_df=df)
    hierarchy = tree.m5_hierarchy
    aggregator = HierarchicalAggregator(hierarchy=hierarchy)
    return {
        "df": df,
        "tree": tree,
        "hierarchy": hierarchy,
        "aggregator": aggregator,
    }


def test_hierarchy_tree_structure(sample_hierarchy):
    """Verifies tree properties, summing matrix dimensions, and parent-child mapping."""
    tree = sample_hierarchy["tree"]
    S, total_names, bottom_names = tree.build_summing_matrix()

    # Total bottom nodes: 5 items x 3 stores = 15 series
    assert len(bottom_names) == 15
    assert S.shape[1] == 15
    assert S.shape[0] == len(total_names)

    # Summing matrix binary entries
    assert set(np.unique(S)).issubset({0.0, 1.0})

    # Each bottom series contributes to exactly one node at each of the 12 levels
    col_sums = S.sum(axis=0)
    assert np.all(col_sums == 12.0)

    # Parent child mapping
    parent_map = tree.get_parent_child_mapping()
    assert "Total" in parent_map
    assert len(parent_map["Total"]) == 2  # 2 states (CA, TX)


def test_aggregation_and_historical_proportions(sample_hierarchy):
    """Verifies bottom-up aggregation and PHA / AHP proportion calculations."""
    aggregator = sample_hierarchy["aggregator"]
    df = sample_hierarchy["df"]

    bottom_wide = aggregator.prepare_bottom_wide_dataframe(df)
    assert bottom_wide.shape[0] == 45  # 45 days
    assert bottom_wide.shape[1] == 15  # 15 bottom series

    # Historical proportions sum to 1.0
    pha_props = aggregator.compute_historical_proportions(bottom_wide, method="pha")
    assert len(pha_props) == 15
    assert np.isclose(pha_props.sum(), 1.0, atol=1e-6)
    assert np.all(pha_props >= 0.0)

    ahp_props = aggregator.compute_historical_proportions(bottom_wide, method="ahp")
    assert len(ahp_props) == 15
    assert np.isclose(ahp_props.sum(), 1.0, atol=1e-6)

    # Top-down disaggregation
    top_demand = np.array([100.0, 200.0])
    disagg = aggregator.disaggregate_top_down(top_demand, pha_props)
    assert disagg.shape == (2, 15)
    assert np.isclose(disagg.sum(axis=1), top_demand, atol=1e-5).all()


def test_coherence_verification(sample_hierarchy):
    """Verifies that coherence checker correctly identifies coherent vs incoherent forecasts."""
    aggregator = sample_hierarchy["aggregator"]
    df = sample_hierarchy["df"]

    bottom_wide = aggregator.prepare_bottom_wide_dataframe(df)
    all_wide = aggregator.aggregate_bottom_to_all_levels(bottom_wide)

    # Perfect coherence
    coh_result = aggregator.verify_coherence(all_wide)
    assert coh_result["is_coherent"] is True
    assert coh_result["max_abs_incoherence"] < 1e-4

    # Inject artificial incoherence into Total node
    incoherent_wide = all_wide.copy()
    incoherent_wide["Total"] += 50.0
    incoh_result = aggregator.verify_coherence(incoherent_wide)
    assert incoh_result["is_coherent"] is False
    assert incoh_result["max_abs_incoherence"] >= 50.0


def test_bottom_up_and_top_down_reconciliation(sample_hierarchy):
    """Verifies BU and TD reconciliation methods produce coherent forecasts."""
    hierarchy = sample_hierarchy["hierarchy"]
    aggregator = sample_hierarchy["aggregator"]
    df = sample_hierarchy["df"]

    bottom_wide = aggregator.prepare_bottom_wide_dataframe(df)
    actual_all = aggregator.aggregate_bottom_to_all_levels(bottom_wide)

    # Create synthetic noisy forecasts
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 2.0, actual_all.shape)
    noisy_forecasts = actual_all.copy() + noise

    # Incoherent initially
    init_coh = aggregator.verify_coherence(noisy_forecasts)
    assert init_coh["is_coherent"] is False

    # 1. Bottom-Up
    bu_rec = HierarchicalReconciler(hierarchy=hierarchy, method="bu", non_negative=False)
    bu_rec.fit()
    reconciled_bu = bu_rec.reconcile(noisy_forecasts)
    coh_bu = aggregator.verify_coherence(reconciled_bu)
    assert coh_bu["is_coherent"] is True

    # 2. Top-Down
    td_rec = HierarchicalReconciler(hierarchy=hierarchy, method="td", non_negative=False)
    td_rec.fit(in_sample_actuals=bottom_wide)
    reconciled_td = td_rec.reconcile(noisy_forecasts)
    coh_td = aggregator.verify_coherence(reconciled_td)
    assert coh_td["is_coherent"] is True


def test_ols_and_mint_reconciliation(sample_hierarchy):
    """Verifies OLS, WLS, and MinT-Shrink reconciliation algorithms."""
    hierarchy = sample_hierarchy["hierarchy"]
    aggregator = sample_hierarchy["aggregator"]
    df = sample_hierarchy["df"]

    bottom_wide = aggregator.prepare_bottom_wide_dataframe(df)
    actual_all = aggregator.aggregate_bottom_to_all_levels(bottom_wide)

    # Generate synthetic in-sample base forecasts
    rng = np.random.RandomState(123)
    in_sample_preds = actual_all.copy() + rng.normal(0, 1.5, actual_all.shape)

    # 1. OLS
    ols_rec = HierarchicalReconciler(hierarchy=hierarchy, method="ols", non_negative=False)
    ols_rec.fit()
    rec_ols = ols_rec.reconcile(in_sample_preds)
    assert aggregator.verify_coherence(rec_ols)["is_coherent"] is True

    # 2. WLS Structural
    wls_rec = HierarchicalReconciler(hierarchy=hierarchy, method="wls_str", non_negative=False)
    wls_rec.fit()
    rec_wls = wls_rec.reconcile(in_sample_preds)
    assert aggregator.verify_coherence(rec_wls)["is_coherent"] is True

    # 3. MinT Shrinkage
    mint_rec = HierarchicalReconciler(hierarchy=hierarchy, method="mint_shrink", non_negative=False)
    mint_rec.fit(in_sample_actuals=actual_all, in_sample_base_forecasts=in_sample_preds)
    rec_mint = mint_rec.reconcile(in_sample_preds)
    assert aggregator.verify_coherence(rec_mint)["is_coherent"] is True


def test_non_negative_reconciliation(sample_hierarchy):
    """Verifies that non-negative reconciliation guarantees both non-negativity and coherence."""
    hierarchy = sample_hierarchy["hierarchy"]
    aggregator = sample_hierarchy["aggregator"]
    df = sample_hierarchy["df"]

    bottom_wide = aggregator.prepare_bottom_wide_dataframe(df)
    actual_all = aggregator.aggregate_bottom_to_all_levels(bottom_wide)

    # Create base forecasts with negative values
    noisy_forecasts = actual_all.copy().iloc[:7] - 50.0  # severely negative

    # Reconcile with non_negative=True
    nn_rec = HierarchicalReconciler(hierarchy=hierarchy, method="ols", non_negative=True)
    nn_rec.fit()
    rec_nn = nn_rec.reconcile(noisy_forecasts)

    # 1. Zero negative values
    assert (rec_nn.values >= -1e-6).all(), "Reconciled forecast contained negative values!"

    # 2. Perfect coherence
    coh_nn = aggregator.verify_coherence(rec_nn)
    assert coh_nn["is_coherent"] is True


def test_reconciliation_impact_evaluation(sample_hierarchy):
    """Verifies before-after reconciliation impact evaluation table."""
    hierarchy = sample_hierarchy["hierarchy"]
    aggregator = sample_hierarchy["aggregator"]
    df = sample_hierarchy["df"]

    bottom_wide = aggregator.prepare_bottom_wide_dataframe(df)
    actual_all = aggregator.aggregate_bottom_to_all_levels(bottom_wide)

    # Base forecasts (noisy)
    rng = np.random.RandomState(99)
    base_fc = actual_all.copy() + rng.normal(0, 3.0, actual_all.shape)

    rec = HierarchicalReconciler(hierarchy=hierarchy, method="mint_shrink", non_negative=False)
    rec.fit(in_sample_actuals=actual_all, in_sample_base_forecasts=base_fc)
    reconciled_fc = rec.reconcile(base_fc)

    impact_df = HierarchicalReconciler.evaluate_reconciliation_impact(
        base_forecasts=base_fc,
        reconciled_forecasts=reconciled_fc,
        actuals=actual_all,
        hierarchy=hierarchy,
    )

    assert not impact_df.empty
    assert "level" in impact_df.columns
    assert "base_wape" in impact_df.columns
    assert "rec_wape" in impact_df.columns
    assert len(impact_df) >= 5


def test_sktime_reconciler_integration(sample_hierarchy):
    """Verifies end-to-end integration with sktime ReconcilerForecaster."""
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.reconcile import ReconcilerForecaster

    aggregator = sample_hierarchy["aggregator"]
    df = sample_hierarchy["df"]

    # Build sktime MultiIndex hierarchical dataframe for state and store levels
    hier_df = aggregator.to_sktime_hierarchical(
        df,
        hierarchy_cols=["state_id", "store_id"],
        date_col="date",
        sales_col="sales",
    )

    forecaster = NaiveForecaster(strategy="last")
    reconciler = ReconcilerForecaster(forecaster, method="ols")
    reconciler.fit(hier_df)
    preds = reconciler.predict(fh=[1, 2])

    assert not preds.empty
    assert "__total" in preds.index.get_level_values("store_id")
