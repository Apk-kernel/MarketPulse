"""Comprehensive unit and integration test suite for MarketPulse research experiments."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.evaluation.experiments import ResearchExperimentRunner
from src.data.loader import M5DataLoader


@pytest.fixture(scope="module")
def processed_m5_data():
    """Provides a synthetic or processed M5 DataFrame for testing."""
    data_path = Path("data/processed/m5_processed.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        loader = M5DataLoader(data_dir=Path("data/raw"), random_seed=42)
        df = loader.generate_synthetic_sample(
            n_items=9,
            n_stores=3,
            n_days=70,
        )
    return df


@pytest.fixture(scope="module")
def experiment_runner(processed_m5_data):
    """Initializes a shared ResearchExperimentRunner fixture."""
    return ResearchExperimentRunner(processed_df=processed_m5_data, horizon=14)


def test_runner_initialization(experiment_runner):
    """Verifies that the runner correctly sets up matrices, indices, and sizes."""
    assert experiment_runner.horizon == 14
    assert experiment_runner.train_size == experiment_runner.T_total - 14
    assert experiment_runner.bottom_wide.shape[1] == 45
    assert experiment_runner.all_wide.shape[1] == 154
    assert len(experiment_runner.train_dates) == experiment_runner.train_size
    assert len(experiment_runner.test_dates) == 14


def test_experiment_1_local_vs_global(experiment_runner):
    """Verifies Experiment 1: Local vs Global ML model execution and schema."""
    df_exp1 = experiment_runner.run_experiment_1_local_vs_global()
    assert isinstance(df_exp1, pd.DataFrame)
    assert len(df_exp1) == 3
    expected_cols = [
        "Evaluation Tier",
        "Local WAPE (%)",
        "Global WAPE (%)",
        "Global Improvement (%)",
        "Local RMSE",
        "Global RMSE",
        "Local Time (s)",
        "Global Time (s)",
    ]
    for col in expected_cols:
        assert col in df_exp1.columns

    for val in df_exp1["Local WAPE (%)"]:
        assert 0.0 < val < 500.0


def test_experiment_2_reconciliation_value(experiment_runner):
    """Verifies Experiment 2: Value of forecast reconciliation under equalized base models."""
    df_exp2 = experiment_runner.run_experiment_2_reconciliation_value()
    assert isinstance(df_exp2, pd.DataFrame)
    assert len(df_exp2) >= 6
    expected_methods = [
        "Base (Unreconciled)",
        "Bottom-Up (BU)",
        "Top-Down (TD-PHA)",
        "Ordinary Least Squares (OLS)",
        "WLS Structural (WLS-Str)",
        "MinT Shrinkage",
        "Non-Negative MinT (NN-MinT)",
    ]
    assert list(df_exp2["Method"]) == expected_methods

    # Base should not be coherent
    base_row = df_exp2[df_exp2["Method"] == "Base (Unreconciled)"].iloc[0]
    assert base_row["Is Coherent"] == "NO"
    assert base_row["Max Incoherence"] > 0.1

    # Reconciled methods must be coherent
    reconciled_rows = df_exp2[df_exp2["Method"] != "Base (Unreconciled)"]
    for _, row in reconciled_rows.iterrows():
        assert row["Is Coherent"] == "YES"
        assert row["Max Incoherence"] <= 1e-3


def test_experiment_3_exogenous_ablation(experiment_runner):
    """Verifies Experiment 3: Exogenous promotional & pricing feature ablation."""
    df_exp3 = experiment_runner.run_experiment_3_exogenous_ablation()
    assert isinstance(df_exp3, pd.DataFrame)
    assert len(df_exp3) == 5
    expected_cols = [
        "Feature Set Configuration",
        "Features Count",
        "Overall WAPE (%)",
        "Overall MAE",
        "SNAP Days WAPE (%)",
        "Marginal Lift (%)",
    ]
    for col in expected_cols:
        assert col in df_exp3.columns

    # Feature count should be non-decreasing
    counts = df_exp3["Features Count"].tolist()
    assert counts == sorted(counts)
    assert counts[0] == 3
    assert counts[-1] == 8


def test_experiment_4_cold_start(experiment_runner):
    """Verifies Experiment 4: Cold-start demand estimation via hierarchical priors."""
    df_exp4 = experiment_runner.run_experiment_4_cold_start()
    assert isinstance(df_exp4, pd.DataFrame)
    assert len(df_exp4) == 3
    assert "Strategy 1: Naive Zero Forecast" in df_exp4["Cold-Start Strategy"].values
    assert "Strategy 2: Category Average Prior" in df_exp4["Cold-Start Strategy"].values
    assert "Strategy 3: Hierarchical Top-Down Prior" in df_exp4["Cold-Start Strategy"].values

    wapes = dict(zip(df_exp4["Cold-Start Strategy"], df_exp4["Cold-Start WAPE (%)"]))
    # Naive zero forecast should have 100% WAPE
    assert wapes["Strategy 1: Naive Zero Forecast"] == 100.0
    # Hierarchical prior should outperform naive zero
    assert wapes["Strategy 3: Hierarchical Top-Down Prior"] < wapes["Strategy 1: Naive Zero Forecast"]


def test_experiment_5_regime_routing(experiment_runner):
    """Verifies Experiment 5: Syntetos-Boylan demand regime routing."""
    df_exp5 = experiment_runner.run_experiment_5_regime_routing()
    assert isinstance(df_exp5, pd.DataFrame)
    assert len(df_exp5) >= 1
    expected_cols = [
        "Demand Regime",
        "Series Count",
        "Universal Model WAPE (%)",
        "Regime-Routed WAPE (%)",
        "Routed Improvement (%)",
    ]
    for col in expected_cols:
        assert col in df_exp5.columns


def test_experiment_6_horizon_deterioration(experiment_runner):
    """Verifies Experiment 6: Forecast horizon deterioration dynamics."""
    df_exp6 = experiment_runner.run_experiment_6_horizon_deterioration()
    assert isinstance(df_exp6, pd.DataFrame)
    assert len(df_exp6) == 4
    expected_horizons = [7, 14, 21, 28]
    assert list(df_exp6["Horizon (H)"]) == expected_horizons
    for col in ["Seasonal Naive WAPE (%)", "Recursive LightGBM WAPE (%)", "LightGBM Advantage (%)"]:
        assert col in df_exp6.columns


def test_multi_origin_rolling_experiment(processed_m5_data):
    """Verifies multi-origin rolling temporal evaluations across multiple forecast cutoffs."""
    runner = ResearchExperimentRunner(
        processed_df=processed_m5_data,
        horizon=14,
        n_origins=2,
        origin_step=14,
    )
    origins = runner.get_rolling_origins()
    assert len(origins) == 2
    assert origins[0]["origin_idx"] == 1
    assert origins[1]["origin_idx"] == 2
    assert origins[0]["train_size"] < origins[1]["train_size"]
    assert origins[0]["test_end_idx"] <= origins[1]["train_size"] + runner.horizon

    # Execute rolling benchmark
    multi_df = runner.run_multi_origin_experiment("run_experiment_1_local_vs_global")
    assert not multi_df.empty
    assert set(multi_df["origin_idx"].unique()) == {1, 2}
    assert "Evaluation Tier" in multi_df.columns
    assert "Global WAPE (%)" in multi_df.columns


