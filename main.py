"""MarketPulse: Hierarchical Probabilistic Demand Forecasting Engine."""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logger
from src.utils.config import load_config

logger = setup_logger("MarketPulse.Main")


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketPulse Enterprise Demand Forecasting Engine")
    parser.add_argument("--monitoring", action="store_true", help="Execute Phase 11 Production Monitoring & Drift Detection")
    parser.add_argument("--inventory", action="store_true", help="Execute Phase 10 Inventory Decision & Monte Carlo Simulation")
    parser.add_argument("--experiments", action="store_true", help="Execute Phase 7 Research Experiments Suite")
    args = parser.parse_args()

    config = load_config()
    logger.info(f"Initialized {config.project.name} v{config.project.version}")
    logger.info(f"Project root: {config.project_root}")

    if args.monitoring:
        logger.info("Executing Phase 11 Production Monitoring & Drift Detection...")
        from scripts.run_monitoring import run_monitoring_pipeline
        run_monitoring_pipeline()
        return 0

    if args.inventory:
        logger.info("Executing Phase 10 Inventory Simulation...")
        from scripts.run_inventory_simulation import main as run_inventory
        return run_inventory()

    if args.experiments:
        logger.info("Executing Phase 7 Research Experiments...")
        from scripts.run_research_experiments import main as run_experiments
        return run_experiments()

    logger.info("System is configured.")
    logger.info("Available execution pipelines:")
    logger.info("  - Phase 1 Data Prep:     python scripts/prepare_data.py")
    logger.info("  - Phase 2 EDA:           python scripts/run_eda.py")
    logger.info("  - Phase 3 Baselines:     python scripts/run_baseline_benchmarks.py")
    logger.info("  - Phase 7 Experiments:   python main.py --experiments")
    logger.info("  - Phase 10 Inventory:    python main.py --inventory")
    logger.info("  - Phase 11 Monitoring:   python main.py --monitoring")
    return 0


if __name__ == "__main__":
    sys.exit(main())
