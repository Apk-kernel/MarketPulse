"""Feature engineering and demand analytics modules for MarketPulse."""

from src.features.demand_features import (
    DemandProfile,
    calculate_demand_profile,
    classify_demand_regimes,
)
from src.features.temporal_features import (
    extract_calendar_features,
    calculate_seasonal_indices,
    calculate_event_uplift,
)
from src.features.price_features import (
    calculate_price_features,
    estimate_price_elasticity,
)
from src.features.lag_features import create_lag_features
from src.features.rolling_features import (
    create_rolling_features,
    create_price_dynamics_features,
)

__all__ = [
    "DemandProfile",
    "calculate_demand_profile",
    "classify_demand_regimes",
    "extract_calendar_features",
    "calculate_seasonal_indices",
    "calculate_event_uplift",
    "calculate_price_features",
    "estimate_price_elasticity",
    "create_lag_features",
    "create_rolling_features",
    "create_price_dynamics_features",
]
