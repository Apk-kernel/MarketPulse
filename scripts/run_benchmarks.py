"""Script to execute Phase 4: Temporal Backtesting and M5 Hierarchical Evaluation."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.hierarchy import M5Hierarchy
from src.evaluation.backtesting import TemporalBacktester
from src.evaluation.metrics import (
    root_mean_squared_scaled_error,
    weighted_root_mean_squared_scaled_error,
)
from src.forecasting.baselines import build_all_baseline_forecasters
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("MarketPulse.RunBenchmarks")


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketPulse Temporal Backtesting & Validation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Forecasting horizon for backtesting (e.g. 14 or 28).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of backtesting folds.",
    )
    parser.add_argument(
        "--window-type",
        type=str,
        default="expanding",
        choices=["expanding", "sliding"],
        help="Temporal window type.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "results" / "backtest_comparison.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "reports" / "backtest_comparison.md"),
        help="Output Markdown report path.",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config))

    input_path = config.project_root / config.data.processed_dir / config.data.processed_file
    if not input_path.exists():
        logger.error(f"Processed dataset not found at {input_path}. Run prepare_data.py first.")
        return 1

    logger.info(f"Loading processed demand dataset from {input_path}...")
    df = pd.read_parquet(input_path)

    # Initialize hierarchy
    hierarchy = M5Hierarchy(metadata_df=df)

    # Prepare multi-level series
    lvl1 = hierarchy.aggregate_bottom_up(df, level=1)
    lvl1_series = lvl1.set_index("date")["sales"].asfreq("D").fillna(0.0)

    lvl2 = hierarchy.aggregate_bottom_up(df, level=2)
    state_ca = lvl2[lvl2["series_id"] == "CA"].set_index("date")["sales"].asfreq("D").fillna(0.0)

    lvl4 = hierarchy.aggregate_bottom_up(df, level=4)
    foods_series = lvl4[lvl4["series_id"] == "FOODS"].set_index("date")["sales"].asfreq("D").fillna(0.0)

    benchmark_series = {
        "Level1_Total": lvl1_series,
        "Level2_State_CA": state_ca,
        "Level4_FOODS": foods_series,
    }

    forecasters = build_all_baseline_forecasters(sp=7)

    logger.info(
        f"Initializing TemporalBacktester with {args.n_splits} {args.window_type} folds, "
        f"horizon={args.horizon} days across {len(forecasters)} models..."
    )

    backtester = TemporalBacktester(
        forecasters=forecasters,
        fh=args.horizon,
        n_splits=args.n_splits,
        window_type=args.window_type,
        sp=7,
    )

    backtester.backtest_multiple_series(benchmark_series)
    backtester.save_results(output_csv=args.output_csv, output_md=args.output_md)

    comparison = backtester.get_aggregated_comparison(sort_by="WAPE")

    print("\n================== TEMPORAL BACKTESTING BENCHMARK COMPARISON ==================")
    print(f"Validation Strategy: {args.window_type.capitalize()} Window ({args.n_splits} Folds, H={args.horizon} days)")
    print(comparison.to_markdown(index=False))
    print("================================================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

