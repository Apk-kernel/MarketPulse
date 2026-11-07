"""Script to execute Phase 5: Global vs Local ML Forecasting Research Experiment."""

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
from src.forecasting.baselines import create_theta_forecaster, create_auto_arima
from src.forecasting.global_models import (
    build_ml_reduction_models,
    create_reduction_forecaster,
    prepare_global_panel_dataframe,
)
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("MarketPulse.RunGlobalModels")


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketPulse Global vs Local Forecasting Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[7, 14, 28],
        help="Forecasting horizons to evaluate (e.g. 7 14 28).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=14,
        help="Lag input window length for tabular reduction models.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "results" / "global_vs_local_experiment.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "reports" / "global_forecasting_report.md"),
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

    # Prepare multi-level representative series for experiment
    lvl1 = hierarchy.aggregate_bottom_up(df, level=1)
    lvl1_series = lvl1.set_index("date")["sales"].asfreq("D").fillna(0.0)

    lvl2 = hierarchy.aggregate_bottom_up(df, level=2)
    state_ca = lvl2[lvl2["series_id"] == "CA"].set_index("date")["sales"].asfreq("D").fillna(0.0)
    state_tx = lvl2[lvl2["series_id"] == "TX"].set_index("date")["sales"].asfreq("D").fillna(0.0)

    lvl4 = hierarchy.aggregate_bottom_up(df, level=4)
    foods_series = lvl4[lvl4["series_id"] == "FOODS"].set_index("date")["sales"].asfreq("D").fillna(0.0)
    hobbies_series = lvl4[lvl4["series_id"] == "HOBBIES"].set_index("date")["sales"].asfreq("D").fillna(0.0)

    benchmark_series = {
        "Level1_Total": lvl1_series,
        "Level2_State_CA": state_ca,
        "Level2_State_TX": state_tx,
        "Level4_FOODS": foods_series,
        "Level4_HOBBIES": hobbies_series,
    }

    logger.info(f"Loaded {len(benchmark_series)} benchmark series across hierarchy levels.")

    # Instantiate candidate ML models and statistical benchmarks
    forecasters = build_ml_reduction_models(
        window_length=args.window_length,
        random_state=config.project.random_seed,
    )
    # Add top statistical baselines for research comparison
    forecasters["baseline_theta"] = create_theta_forecaster(sp=7)
    forecasters["baseline_auto_arima"] = create_auto_arima(sp=1)

    logger.info(f"Running benchmark with models: {list(forecasters.keys())} across horizons {args.horizons}...")

    engine = BaselineBenchmarkEngine(
        forecasters=forecasters,
        horizons=args.horizons,
        sp=7,
        fixed_origin=True,
    )

    results_df = engine.evaluate_multiple_series(benchmark_series)
    engine.save_results(output_csv_path=args.output_csv, output_leaderboard_md=args.output_md)

    leaderboard = engine.get_leaderboard(sort_by="WAPE")

    print("\n================== GLOBAL VS LOCAL FORECASTING EXPERIMENT ==================")
    print(f"Lag Input Window: {args.window_length} days | Evaluation: Fixed Origin Multi-Horizon")
    print(leaderboard.to_markdown(index=False))
    print("============================================================================\n")

    logger.info("Global vs Local forecasting experiment completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

