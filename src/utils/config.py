"""Configuration management for MarketPulse."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ProjectConfig:
    name: str = "MarketPulse"
    version: str = "0.1.0"
    random_seed: int = 42


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    sales_file: str = "sales_train_validation.csv"
    calendar_file: str = "calendar.csv"
    prices_file: str = "sell_prices.csv"
    processed_file: str = "m5_processed.parquet"
    processed_sample_file: str = "m5_processed_sample.parquet"
    quality_report_file: str = "data_quality_report.json"
    chunk_size: int = 100000
    downcast_dtypes: bool = True


@dataclass
class ValidationConfig:
    allow_negative_sales: bool = False
    enforce_date_continuity: bool = True
    enforce_hierarchy_consistency: bool = True
    flag_unlisted_as_nan: bool = True


@dataclass
class HierarchyConfig:
    structure: List[str] = field(
        default_factory=lambda: ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
    )
    aggregation_levels: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "level_1": ["state_id"],
            "level_2": ["state_id", "store_id"],
            "level_3": ["state_id", "store_id", "cat_id"],
            "level_4": ["state_id", "store_id", "dept_id"],
            "level_5": ["state_id", "store_id", "item_id"],
        }
    )


@dataclass
class Config:
    project_root: Path
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    forecasting: Dict[str, Any] = field(default_factory=dict)

    def get_raw_path(self, filename: str) -> Path:
        return self.project_root / self.data.raw_dir / filename

    def get_processed_path(self, filename: str) -> Path:
        return self.project_root / self.data.processed_dir / filename


def load_config(config_path: Optional[Path] = None) -> Config:
    """Loads configuration from YAML file and constructs Config dataclass.

    Args:
        config_path: Optional path to config.yaml. Defaults to configs/config.yaml
                     relative to the repository root.

    Returns:
        Config instance.
    """
    if config_path is None:
        # Default project root is 2 levels up from this file (src/utils/config.py -> MarketPulse)
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"
    else:
        project_root = config_path.resolve().parent.parent

    if not config_path.exists():
        return Config(project_root=project_root)

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    project_data = raw_dict.get("project", {})
    data_data = raw_dict.get("data", {})
    validation_data = raw_dict.get("validation", {})
    hierarchy_data = raw_dict.get("hierarchy", {})
    forecasting_data = raw_dict.get("forecasting", {})

    return Config(
        project_root=project_root,
        project=ProjectConfig(**project_data),
        data=DataConfig(**data_data),
        validation=ValidationConfig(**validation_data),
        hierarchy=HierarchyConfig(**hierarchy_data),
        forecasting=forecasting_data,
    )

