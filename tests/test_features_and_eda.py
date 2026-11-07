"""Unit and Integration Tests for Demand Analytics, Feature Engineering, and EDA Reporting."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.loader import M5DataLoader
from src.data.hierarchy import M5Hierarchy
from src.features.demand_features import calculate_demand_profile, classify_demand_regimes
from src.features.temporal_features import (
    extract_calendar_features,
    calculate_seasonal_indices,
    calculate_event_uplift,
)
from src.features.price_features import calculate_price_features, estimate_price_elasticity
from src.evaluation.reporting import EDAReporter


def test_calculate_demand_profile_known_regimes():
    """Tests Syntetos-Boylan ADI and CV^2 calculation against analytically known patterns."""
    # 1. Smooth: non-zero every day (ADI = 1.0 < 1.32), constant/low variance (CV^2 ~ 0 < 0.49)
    smooth_series = pd.Series([5, 5, 6, 5, 5, 6, 5, 5, 6, 5])
    prof_smooth = calculate_demand_profile(smooth_series, "smooth_test")
    assert prof_smooth.regime == "Smooth"
    assert prof_smooth.adi == 1.0
    assert prof_smooth.cv2 < 0.49

    # 2. Intermittent: demand every 3 days (ADI = 3.0 >= 1.32), constant quantity (CV^2 = 0 < 0.49)
    intermittent_series = pd.Series([0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5])
    prof_intermittent = calculate_demand_profile(intermittent_series, "intermittent_test")
    assert prof_intermittent.regime == "Intermittent"
    assert prof_intermittent.adi >= 1.32
    assert prof_intermittent.cv2 < 0.49

    # 3. Erratic: non-zero every day (ADI = 1.0 < 1.32), highly volatile order size (CV^2 >= 0.49)
    erratic_series = pd.Series([1, 20, 2, 25, 1, 30, 2, 18])
    prof_erratic = calculate_demand_profile(erratic_series, "erratic_test")
    assert prof_erratic.regime == "Erratic"
    assert prof_erratic.adi < 1.32
    assert prof_erratic.cv2 >= 0.49

    # 4. Lumpy: sporadic occurrence (ADI >= 1.32) and volatile sizes (CV^2 >= 0.49)
    lumpy_series = pd.Series([0, 0, 1, 0, 0, 0, 35, 0, 0, 2, 0, 0, 40])
    prof_lumpy = calculate_demand_profile(lumpy_series, "lumpy_test")
    assert prof_lumpy.regime == "Lumpy"
    assert prof_lumpy.adi >= 1.32
    assert prof_lumpy.cv2 >= 0.49


def test_classify_demand_regimes_dataframe():
    """Tests batch classification of multi-series dataframe."""
    df = pd.DataFrame({
        "store_id": ["CA_1"] * 20 + ["CA_2"] * 20,
        "item_id": ["item_1"] * 20 + ["item_2"] * 20,
        "sales": [5.0] * 20 + [0.0, 0.0, 10.0, 0.0] * 5,
    })
    regimes = classify_demand_regimes(df, id_cols=["store_id", "item_id"])
    assert len(regimes) == 2
    assert "regime" in regimes.columns
    assert "adi" in regimes.columns
    assert "cv2" in regimes.columns


def test_extract_calendar_features():
    """Tests temporal feature extraction and cyclical encoding bounds."""
    dates = pd.date_range("2021-01-01", periods=14, freq="D")
    df = pd.DataFrame({"date": dates, "sales": range(14)})
    enriched = extract_calendar_features(df, date_col="date", add_cyclical=True)

    assert "dayofweek" in enriched.columns
    assert "month" in enriched.columns
    assert "is_weekend" in enriched.columns
    assert "dow_sin" in enriched.columns
    assert "dow_cos" in enriched.columns

    # Check bounds [-1, 1] for cyclical trigonometric features
    assert enriched["dow_sin"].between(-1.0, 1.0).all()
    assert enriched["dow_cos"].between(-1.0, 1.0).all()


def test_seasonal_indices_and_uplift():
    """Tests seasonal index calculation and event uplift ratios."""
    dates = pd.date_range("2021-01-01", periods=28, freq="D")
    # Simulate higher weekend sales
    sales = [10.0 if d.dayofweek in [5, 6] else 5.0 for d in dates]
    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "event_name_1": ["SuperBowl" if d.day == 5 else None for d in dates],
        "snap_active": [1 if d.day <= 10 else 0 for d in dates],
    })

    indices = calculate_seasonal_indices(df)
    dow = indices["dow_profile"]
    # Saturday and Sunday must have higher seasonal index than Monday
    sat_idx = dow.loc[dow["day_name"] == "Saturday", "seasonal_index"].values[0]
    mon_idx = dow.loc[dow["day_name"] == "Monday", "seasonal_index"].values[0]
    assert sat_idx > mon_idx

    # Event uplift
    uplift = calculate_event_uplift(df)
    assert "events" in uplift
    assert "snap" in uplift
    assert "snap_uplift_ratio" in uplift["snap"]


def test_price_features_and_elasticity():
    """Tests discount depth calculation and log-log price elasticity estimation."""
    # Synthetic demand where quantity drops as price increases: Q = 100 / P
    prices = [2.0, 4.0, 5.0, 10.0] * 5
    sales = [round(100.0 / p) for p in prices]
    df = pd.DataFrame({
        "store_id": ["CA_1"] * len(prices),
        "item_id": ["item_1"] * len(prices),
        "cat_id": ["FOODS"] * len(prices),
        "sell_price": prices,
        "sales": sales,
    })

    df_price = calculate_price_features(df)
    assert "discount_depth" in df_price.columns
    assert "is_discounted" in df_price.columns
    assert (df_price["discount_depth"] >= 0.0).all()

    # Elasticity regression
    elasticity_df = estimate_price_elasticity(df_price, group_col="cat_id")
    assert not elasticity_df.empty
    beta = elasticity_df.loc[elasticity_df["group"] == "FOODS", "elasticity"].values[0]
    # Price elasticity must be strictly negative (downward sloping demand)
    assert beta < 0.0


def test_price_features_expanding_max_no_future_leakage():
    """Validates that base_price uses causal expanding cummax, preventing future price lookahead."""
    dates = pd.date_range("2021-01-01", periods=6, freq="D")
    # Low price initially ($2), then price hike ($5), then slight promotion ($4)
    df = pd.DataFrame({
        "date": dates,
        "store_id": ["CA_1"] * 6,
        "item_id": ["item_1"] * 6,
        "sell_price": [2.0, 2.0, 2.0, 5.0, 5.0, 4.0],
    })

    # Shuffle row order to verify robust chronological index alignment
    shuffled = df.sample(frac=1.0, random_state=42)
    result = calculate_price_features(shuffled, date_col="date", use_expanding_max=True)
    res_sorted = result.sort_values("date").reset_index(drop=True)

    # Days 0, 1, 2: base_price should be 2.0, discount_depth = 0.0 (NO future $5 leakage)
    assert np.allclose(res_sorted.loc[0:2, "base_price"], 2.0)
    assert np.allclose(res_sorted.loc[0:2, "discount_depth"], 0.0)
    assert (res_sorted.loc[0:2, "is_discounted"] == 0).all()

    # Days 3, 4: base_price expands to 5.0, discount_depth = 0.0
    assert np.allclose(res_sorted.loc[3:4, "base_price"], 5.0)
    assert np.allclose(res_sorted.loc[3:4, "discount_depth"], 0.0)

    # Day 5: price dropped to $4.00 against base $5.00 -> 20% discount
    assert np.isclose(res_sorted.loc[5, "base_price"], 5.0)
    assert np.isclose(res_sorted.loc[5, "discount_depth"], 0.20)
    assert res_sorted.loc[5, "is_discounted"] == 1


def test_full_eda_reporter_artifacts():
    """Integration test: verifies EDAReporter generates all 5 publication plots and markdown/json reports."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_path = tmp_path / "raw"
        out_reports = tmp_path / "reports"

        # Generate sample data
        M5DataLoader.generate_synthetic_sample(
            output_dir=raw_path,
            n_items=8,
            n_stores=2,
            n_days=45,
            random_seed=42,
        )

        from src.data.preprocessing import M5Preprocessor
        loader = M5DataLoader(raw_path)
        sales = loader.load_sales_raw()
        cal = loader.load_calendar()
        prices = loader.load_prices()

        preprocessor = M5Preprocessor()
        processed, _ = preprocessor.process(sales, cal, prices)

        hierarchy = M5Hierarchy(metadata_df=sales)
        reporter = EDAReporter(output_dir=out_reports)

        summary = reporter.run_full_eda(df=processed, hierarchy=hierarchy)

        # Check summary structure
        assert "A_sales_distribution" in summary
        assert "B_temporal_behaviour" in summary
        assert "C_product_behaviour" in summary
        assert "D_price_behaviour" in summary
        assert "E_hierarchical_analysis" in summary

        # Check saved files
        assert (out_reports / "eda_summary.json").is_file()
        assert (out_reports / "eda_summary.md").is_file()

        # Check all 5 figures were generated
        fig_dir = out_reports / "figures"
        assert (fig_dir / "01_sales_distribution.png").is_file()
        assert (fig_dir / "02_temporal_behaviour.png").is_file()
        assert (fig_dir / "03_demand_regimes_scatter.png").is_file()
        assert (fig_dir / "04_price_behaviour.png").is_file()
        assert (fig_dir / "05_hierarchical_aggregations.png").is_file()

