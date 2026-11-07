"""Script to execute Phase 3: Baseline Forecasting & Multi-Horizon Benchmarking."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.hierarchy import M5Hierarchy
from src.evaluation.benchmarking import BaselineBenchmarkEngine
from src.forecasting.baselines import build_all_baseline_forecasters
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("MarketPulse.RunBaselines")


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketPulse Baseline Forecasting Benchmark")
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
        "--horizons",
        nargs="+",
        type=int,
        default=[7, 14, 28],
        help="Forecast horizons to evaluate (e.g. 7 14 28).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "results" / "baseline_benchmarks.csv"),
        help="Path to save detailed evaluation results CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "reports" / "baseline_leaderboard.md"),
        help="Path to save leaderboard markdown report.",
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

    logger.info(f"Loading processed demand from {input_path}...")
    df = pd.read_parquet(input_path)

    # Initialize hierarchy to extract hierarchical series for benchmarking
    hierarchy = M5Hierarchy(metadata_df=df)

    # Prepare benchmark series:
    # 1. Level 1: Total Demand
    lvl1 = hierarchy.aggregate_bottom_up(df, level=1)
    lvl1_series = lvl1.set_index("date")["sales"].asfreq("D").fillna(0.0)

    # 2. Level 2: State Demand (Sample CA and TX)
    lvl2 = hierarchy.aggregate_bottom_up(df, level=2)
    state_series_dict = {}
    for st in lvl2["series_id"].unique()[:2]:
        sub = lvl2[lvl2["series_id"] == st].set_index("date")["sales"].asfreq("D").fillna(0.0)
        state_series_dict[f"State_{st}"] = sub

    # 3. Level 4: Category Demand (FOODS)
    lvl4 = hierarchy.aggregate_bottom_up(df, level=4)
    foods_sub = lvl4[lvl4["series_id"] == "FOODS"].set_index("date")["sales"].asfreq("D").fillna(0.0)

    # 4. Level 12: Sample SKU bottom series
    first_store = df["store_id"].unique()[0]
    first_item = df["item_id"].unique()[0]
    bottom_sub = (
        df[(df["store_id"] == first_store) & (df["item_id"] == first_item)]
        .set_index("date")["sales"]
        .asfreq("D")
        .fillna(0.0)
    )

    benchmark_series = {
        "Level1_Total": lvl1_series,
        "Level4_FOODS": foods_sub,
        f"Level12_{first_store}_{first_item}": bottom_sub,
    }
    benchmark_series.update(state_series_dict)

    logger.info(f"Selected {len(benchmark_series)} benchmark series across hierarchy levels.")

    # Instantiate all sktime baseline forecasters
    forecasters = build_all_baseline_forecasters(sp=7)
    logger.info(f"Loaded {len(forecasters)} baseline forecasters: {list(forecasters.keys())}")

    # Initialize benchmark engine
    engine = BaselineBenchmarkEngine(
        forecasters=forecasters,
        horizons=args.horizons,
        sp=7,
    )

    # Run benchmark evaluation
    results_df = engine.evaluate_multiple_series(benchmark_series)
    engine.save_results(output_csv_path=args.output_csv, output_leaderboard_md=args.output_md)

    leaderboard = engine.get_leaderboard(sort_by="WAPE")
    print("\n=================== BASELINE FORECASTING BENCHMARK LEADERBOARD ===================")
    print(leaderboard.to_markdown(index=False))
    print("==================================================================================\n")

    logger.info("Baseline forecasting benchmark completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

