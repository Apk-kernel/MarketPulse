"""Comprehensive Unit & Integration Test Suite for Data Engineering Pipeline."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.loader import M5DataLoader
from src.data.validation import DataValidator, DataQualityReport
from src.data.preprocessing import M5Preprocessor
from src.data.hierarchy import M5Hierarchy


@pytest.fixture
def sample_raw_dir():
    """Generates a temporary synthetic M5 raw dataset for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        M5DataLoader.generate_synthetic_sample(
            output_dir=tmp_path,
            n_items=6,
            n_stores=2,
            n_days=30,
            random_seed=123,
        )
        yield tmp_path


def test_data_loader_file_checks(sample_raw_dir):
    """Tests file existence checks in M5DataLoader."""
    loader = M5DataLoader(sample_raw_dir)
    status = loader.check_raw_files_exist()
    assert all(status.values())
    assert len(loader.get_missing_files()) == 0

    # Test missing files detection
    empty_loader = M5DataLoader(sample_raw_dir / "non_existent")
    assert len(empty_loader.get_missing_files()) == 3


def test_data_loader_load(sample_raw_dir):
    """Tests loading raw dataframes with expected schemas."""
    loader = M5DataLoader(sample_raw_dir)
    cal = loader.load_calendar()
    prices = loader.load_prices()
    sales = loader.load_sales_raw()

    assert "date" in cal.columns
    assert "sell_price" in prices.columns
    assert "id" in sales.columns
    assert any(c.startswith("d_") for c in sales.columns)
    assert len(sales) == 12  # 6 items * 2 stores


def test_schema_validation_failures():
    """Tests that missing required columns raise descriptive ValueErrors."""
    invalid_sales = pd.DataFrame({"id": ["item1"], "item_id": ["item1"]})
    with pytest.raises(ValueError, match="Raw sales missing required columns"):
        DataValidator.validate_raw_sales_schema(invalid_sales)

    invalid_cal = pd.DataFrame({"date": ["2011-01-29"]})
    with pytest.raises(ValueError, match="Calendar missing required columns"):
        DataValidator.validate_calendar_schema(invalid_cal)

    invalid_prices = pd.DataFrame({"store_id": ["CA_1"]})
    with pytest.raises(ValueError, match="Prices missing required columns"):
        DataValidator.validate_prices_schema(invalid_prices)


def test_melt_wide_to_long(sample_raw_dir):
    """Tests wide-to-long transformation preserves all information."""
    loader = M5DataLoader(sample_raw_dir)
    sales = loader.load_sales_raw()

    preprocessor = M5Preprocessor()
    melted = preprocessor.melt_sales_wide_to_long(sales)

    d_cols = [c for c in sales.columns if c.startswith("d_")]
    expected_rows = len(sales) * len(d_cols)

    assert len(melted) == expected_rows
    assert "sales" in melted.columns
    assert "d" in melted.columns
    assert set(sales["id"]).issubset(set(melted["id"]))
    # Total sales volume must remain invariant under reshape
    raw_total = sales[d_cols].sum().sum()
    melted_total = melted["sales"].sum()
    assert np.isclose(raw_total, melted_total)


def test_zero_sales_vs_missing_unlisted(sample_raw_dir):
    """Tests that active zero sales are kept distinct from unlisted/missing items."""
    loader = M5DataLoader(sample_raw_dir)
    sales = loader.load_sales_raw()
    cal = loader.load_calendar()
    prices = loader.load_prices()

    # Preprocessor with flag_unlisted_as_nan=True
    preprocessor = M5Preprocessor(flag_unlisted_as_nan=True)
    melted = preprocessor.melt_sales_wide_to_long(sales)
    merged = preprocessor.merge_calendar_and_prices(melted, cal, prices)

    # Active items should have sell_price notna
    assert "is_active" in merged.columns
    assert "sell_price" in merged.columns

    # Verify zero sales are counted in report without treating active zeroes as missing NaNs
    report = DataValidator.audit_processed_dataframe(merged)
    assert report.total_records == len(merged)
    assert report.zero_sales_count > 0


def test_duplicate_and_continuity_detection():
    """Tests that duplicate rows and date gaps are caught by DataValidator."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": list(dates) + [dates[0]],  # Duplicate first date
        "item_id": ["item1"] * 6,
        "dept_id": ["dept1"] * 6,
        "cat_id": ["cat1"] * 6,
        "store_id": ["store1"] * 6,
        "state_id": ["state1"] * 6,
        "sales": [1.0, 2.0, 3.0, 0.0, 5.0, 1.0],
        "sell_price": [2.5] * 6,
        "weekday": ["Wednesday"] * 6,
        "month": [1] * 6,
        "year": [2020] * 6,
    })

    report = DataValidator.audit_processed_dataframe(df)
    assert report.duplicate_records_count > 0
    assert not report.is_valid

    # Discontinuity test (gap in dates)
    gap_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")]  # Missing Jan 2
    df_gap = pd.DataFrame({
        "date": gap_dates,
        "item_id": ["item1", "item1"],
        "dept_id": ["dept1", "dept1"],
        "cat_id": ["cat1", "cat1"],
        "store_id": ["store1", "store1"],
        "state_id": ["state1", "state1"],
        "sales": [1.0, 2.0],
        "sell_price": [2.5, 2.5],
        "weekday": ["Wednesday", "Friday"],
        "month": [1, 1],
        "year": [2020, 2020],
    })
    report_gap = DataValidator.audit_processed_dataframe(df_gap, enforce_continuity=True)
    assert report_gap.date_continuity_violations > 0
    assert not report_gap.is_valid


def test_hierarchy_tree_properties_and_summing_matrix(sample_raw_dir):
    """Tests M5Hierarchy tree validation, level node retrieval, and summing matrix construction."""
    loader = M5DataLoader(sample_raw_dir)
    sales = loader.load_sales_raw()

    hierarchy = M5Hierarchy(metadata_df=sales)
    tree_status = hierarchy.validate_strict_tree_property()
    assert all(tree_status.values())

    # Check level nodes
    states = hierarchy.get_level_nodes(level=2)
    assert len(states) >= 1
    assert "Total" in hierarchy.get_level_nodes(level=1)

    # Check summing matrix S
    s_matrix, all_names, bottom_names = hierarchy.build_summing_matrix()
    assert s_matrix.shape[1] == len(sales)
    assert s_matrix.shape[0] == len(all_names)

    # Top row (Total) must sum all bottom series (all ones in row 0)
    total_row = s_matrix[0].toarray().flatten()
    assert (total_row == 1.0).all()


def test_end_to_end_pipeline(sample_raw_dir):
    """Tests the full preprocessing pipeline and Parquet export."""
    loader = M5DataLoader(sample_raw_dir)
    cal = loader.load_calendar()
    prices = loader.load_prices()
    sales = loader.load_sales_raw()

    preprocessor = M5Preprocessor(downcast_dtypes=True)
    processed, report = preprocessor.process(sales, cal, prices)

    assert report.is_valid
    assert len(report.issues) == 0
    assert "sales" in processed.columns
    assert "sell_price" in processed.columns
    assert "date" in processed.columns

    # Test Parquet export
    out_parquet = sample_raw_dir / "processed_test.parquet"
    saved_path = M5Preprocessor.save_processed(processed, out_parquet, format="parquet")
    assert saved_path.is_file()

    # Read back and verify
    reloaded = pd.read_parquet(saved_path)
    assert len(reloaded) == len(processed)

