"""Script to execute Phase 2: Exploratory Data Analysis & Publication Visualizations."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.hierarchy import M5Hierarchy
from src.evaluation.reporting import EDAReporter
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("MarketPulse.RunEDA")


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketPulse Exploratory Demand Analysis Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to processed parquet file. Defaults to config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "reports"),
        help="Output directory for reports and figures.",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config))

    input_path = (
        Path(args.input)
        if args.input
        else config.project_root / config.data.processed_dir / config.data.processed_file
    )

    if not input_path.exists():
        logger.error(
            f"Processed data file not found at {input_path}. "
            "Please run `python scripts/prepare_data.py` first."
        )
        return 1

    logger.info(f"Loading processed dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} records for {df['item_id'].nunique()} items and {df['store_id'].nunique()} stores.")

    # Initialize hierarchy
    hierarchy = M5Hierarchy(metadata_df=df)

    # Initialize reporter and run full EDA
    reporter = EDAReporter(output_dir=Path(args.output_dir))
    summary = reporter.run_full_eda(df=df, hierarchy=hierarchy)

    logger.info(
        f"EDA complete. Generated 5 publication figures in {reporter.figures_dir} "
        f"and summary report in {args.output_dir}."
    )

    # Read and print the markdown summary
    md_report_path = Path(args.output_dir) / "eda_summary.md"
    if md_report_path.exists():
        with open(md_report_path, "r", encoding="utf-8") as f:
            print("\n" + f.read() + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

