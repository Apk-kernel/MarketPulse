"""Script to execute Phase 1: Data Ingestion, Preprocessing, and Quality Audit."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import M5DataLoader
from src.data.preprocessing import M5Preprocessor
from src.data.validation import DataValidator
from src.data.hierarchy import M5Hierarchy
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("MarketPulse.PrepareData")


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketPulse Data Preparation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic M5-compatible sample dataset if raw files are not found.",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=20,
        help="Number of SKUs for demo sample generation.",
    )
    parser.add_argument(
        "--n-stores",
        type=int,
        default=3,
        help="Number of stores for demo sample generation.",
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=180,
        help="Number of days for demo sample generation.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output format for processed dataset.",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config))

    raw_dir = config.project_root / config.data.raw_dir
    processed_dir = config.project_root / config.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    loader = M5DataLoader(raw_dir=raw_dir)
    missing = loader.get_missing_files()

    if missing:
        logger.warning(f"Raw M5 files missing in {raw_dir}: {missing}")
        if args.demo:
            logger.info("Demo mode enabled. Generating synthetic M5 dataset...")
            loader.generate_synthetic_sample(
                output_dir=raw_dir,
                n_items=args.n_items,
                n_stores=args.n_stores,
                n_days=args.n_days,
                random_seed=config.project.random_seed,
            )
        else:
            logger.error(
                "\n"
                "========================================================================\n"
                "DATASET NOT FOUND:\n"
                "Please download the M5 Forecasting - Accuracy competition dataset:\n"
                "https://www.kaggle.com/c/m5-forecasting-accuracy/data\n"
                "Place the following uncompressed CSVs into:\n"
                f"  {raw_dir}\n"
                "  - sales_train_validation.csv\n"
                "  - calendar.csv\n"
                "  - sell_prices.csv\n"
                "\n"
                "Alternatively, run this script with --demo to generate a schema-perfect\n"
                "synthetic M5 dataset for development and automated testing:\n"
                "  python scripts/prepare_data.py --demo\n"
                "========================================================================"
            )
            return 1

    # Load raw data
    logger.info("Loading raw M5 datasets...")
    calendar_df = loader.load_calendar()
    prices_df = loader.load_prices()
    sales_df = loader.load_sales_raw()

    # Preprocessing
    preprocessor = M5Preprocessor(
        downcast_dtypes=config.data.downcast_dtypes,
        flag_unlisted_as_nan=config.validation.flag_unlisted_as_nan,
    )
    processed_df, report = preprocessor.process(
        sales_df=sales_df,
        calendar_df=calendar_df,
        prices_df=prices_df,
    )

    # Save processed dataset
    output_filename = (
        config.data.processed_file
        if args.format == "parquet"
        else config.data.processed_file.replace(".parquet", ".csv")
    )
    output_path = processed_dir / output_filename
    preprocessor.save_processed(processed_df, output_path, format=args.format)

    # Save Data Quality Reports
    json_report_path = processed_dir / config.data.quality_report_file
    report.save_json(json_report_path)

    md_report_path = processed_dir / "data_quality_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    logger.info(f"Saved Markdown Data Quality Report to {md_report_path}")

    # Hierarchy verification
    hierarchy = M5Hierarchy(metadata_df=sales_df)
    tree_status = hierarchy.validate_strict_tree_property()
    logger.info(f"Hierarchy verification: {tree_status}")

    s_matrix, all_nodes, bottom_nodes = hierarchy.build_summing_matrix()
    logger.info(
        f"Summing matrix verified: {s_matrix.shape[0]} aggregate series, "
        f"{s_matrix.shape[1]} bottom series."
    )

    print("\n" + report.to_markdown() + "\n")
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())

