"""Automated Model Selection Engine for MarketPulse.

Provides:
1. Cross-validated evaluation across diverse forecasting paradigms:
   - Statistical baselines (Seasonal Naive, Naive Last)
   - Classical decomposition / smoothing (Theta, Exponential Smoothing)
   - Supervised ML reduction (LightGBM)
   - Specialized intermittent models (Croston)
2. Demand-regime aware candidate filtering (Syntetos-Boylan criteria).
3. Zero lookahead temporal cross-validation.
4. Champion model determination and JSON configuration serialization.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from sktime.forecasting.croston import Croston

from src.features.demand_features import calculate_demand_profile
from src.forecasting.base import DemandForecasterWrapper
from src.forecasting.baselines import (
    create_exponential_smoothing,
    create_naive_last,
    create_seasonal_naive,
    create_theta_forecaster,
)
from src.forecasting.global_models import create_reduction_forecaster
from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Optimization.ModelSelector")


@dataclass
class ModelSelectionConfig:
    """Configuration container for automated model selection."""

    horizon: int = 14
    n_splits: int = 2
    metric: str = "wape"  # "wape", "rmse", "mae"
    use_regime_routing: bool = True
    candidate_names: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSelectionResult:
    """Evaluation result for a single series after candidate model benchmarking."""

    series_id: str
    regime: str
    adi: float
    cv2: float
    champion_model: str
    champion_score: float
    all_scores: Dict[str, float]
    runner_up_model: Optional[str] = None
    runner_up_score: Optional[float] = None
    selection_margin: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoModelSelector:
    """Automated model selection and per-series forecasting routing engine."""

    def __init__(self, config: Optional[ModelSelectionConfig] = None):
        """Initializes model selector with configuration and candidate model factories."""
        self.config = config or ModelSelectionConfig()
        self.factories: Dict[str, Callable[[], DemandForecasterWrapper]] = {}
        self.selection_results: Dict[str, ModelSelectionResult] = {}
        self.routing_map: Dict[str, str] = {}

        self._register_default_candidates()

    def _register_default_candidates(self) -> None:
        """Registers default suite of candidate model factories."""
        self.register_candidate("Seasonal_Naive", lambda: create_seasonal_naive(sp=7))
        self.register_candidate("Naive_Last", lambda: create_naive_last())
        self.register_candidate("Theta", lambda: create_theta_forecaster(sp=7))
        self.register_candidate(
            "Exponential_Smoothing", lambda: create_exponential_smoothing(sp=7)
        )
        self.register_candidate(
            "LightGBM_Reduction",
            lambda: create_reduction_forecaster(
                window_length=min(14, self.config.horizon), strategy="recursive"
            ),
        )
        self.register_candidate(
            "Croston",
            lambda: DemandForecasterWrapper(name="Croston", estimator=Croston()),
        )

    def register_candidate(
        self, name: str, factory: Callable[[], DemandForecasterWrapper]
    ) -> None:
        """Registers a named candidate model factory."""
        self.factories[name] = factory

    def _get_active_candidates(self, regime: str) -> List[str]:
        """Filters candidate models based on demand regime if routing is enabled."""
        all_candidates = (
            self.config.candidate_names
            if self.config.candidate_names is not None
            else list(self.factories.keys())
        )

        if not self.config.use_regime_routing:
            return all_candidates

        # Syntetos-Boylan regime filtering
        if regime in ("Intermittent", "Lumpy"):
            # Intermittent slow-movers benefit from Croston, Naive, Seasonal Naive, and Theta
            preferred = ["Croston", "Seasonal_Naive", "Naive_Last", "Theta"]
            active = [
                c
                for c in all_candidates
                if c in preferred
                or "croston" in c.lower()
                or "intermittent" in c.lower()
            ]
            return active if active else all_candidates
        else:
            # Smooth and Erratic fast-movers benefit from ML, Tuned ML, Theta, and Holt-Winters
            preferred = [
                "Tuned_LightGBM",
                "LightGBM_Reduction",
                "Theta",
                "Exponential_Smoothing",
                "Seasonal_Naive",
            ]
            active = [
                c
                for c in all_candidates
                if c in preferred
                or "lightgbm" in c.lower()
                or "lgb" in c.lower()
            ]
            return active if active else all_candidates

    def _evaluate_candidate_on_split(
        self,
        forecaster: DemandForecasterWrapper,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> float:
        """Fits forecaster on y_train and evaluates out-of-time accuracy on y_test."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.fit(y_train)
            fh = np.arange(1, len(y_test) + 1)
            preds = forecaster.predict(fh=fh)

        pred_vals = np.maximum(0.0, preds.values.ravel())
        act_vals = y_test.values.ravel()

        if self.config.metric == "rmse":
            return float(root_mean_squared_error(act_vals, pred_vals))
        elif self.config.metric == "mae":
            return float(mean_absolute_error(act_vals, pred_vals))
        else:  # WAPE
            return float(weighted_absolute_percentage_error(act_vals, pred_vals))

    def evaluate_series(
        self, series: pd.Series, series_id: str
    ) -> ModelSelectionResult:
        """Benchmarking across candidate models for a single series over temporal folds."""
        profile = calculate_demand_profile(series, series_id=series_id)
        regime = profile.regime
        active_candidates = self._get_active_candidates(regime)

        T = len(series)
        fh = self.config.horizon
        n_splits = self.config.n_splits
        if T < (n_splits + 1) * fh:
            n_splits = 1

        candidate_scores: Dict[str, float] = {}

        for c_name in active_candidates:
            factory = self.factories[c_name]
            fold_scores = []
            for fold in range(n_splits):
                test_start = T - (n_splits - fold) * fh
                test_end = test_start + fh

                y_tr = series.iloc[:test_start]
                y_te = series.iloc[test_start:test_end]

                try:
                    forecaster = factory()
                    score = self._evaluate_candidate_on_split(forecaster, y_tr, y_te)
                    fold_scores.append(score)
                except Exception as e:
                    logger.debug(f"Candidate {c_name} failed on fold {fold} for {series_id}: {e}")
                    fold_scores.append(999.0)

            candidate_scores[c_name] = round(float(np.mean(fold_scores)), 3)

        # Rank candidates by score ascending
        sorted_candidates = sorted(candidate_scores.items(), key=lambda kv: kv[1])
        champion_name, champion_score = sorted_candidates[0]

        runner_up_name = sorted_candidates[1][0] if len(sorted_candidates) > 1 else None
        runner_up_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else None
        margin = (
            round(runner_up_score - champion_score, 3)
            if runner_up_score is not None
            else 0.0
        )

        res = ModelSelectionResult(
            series_id=series_id,
            regime=regime,
            adi=round(profile.adi, 2),
            cv2=round(profile.cv2, 2),
            champion_model=champion_name,
            champion_score=champion_score,
            all_scores=candidate_scores,
            runner_up_model=runner_up_name,
            runner_up_score=runner_up_score,
            selection_margin=margin,
        )

        self.selection_results[series_id] = res
        self.routing_map[series_id] = champion_name
        return res

    def select_models_for_portfolio(
        self,
        wide_df: pd.DataFrame,
    ) -> Tuple[Dict[str, str], pd.DataFrame]:
        """Runs automated model selection across all series in a wide DataFrame.

        Args:
            wide_df: DataFrame with DatetimeIndex and series columns.

        Returns:
            Tuple of:
            - routing_map: Dict mapping series_id to champion model name.
            - leaderboard_df: Summary DataFrame with per-series selection metrics.
        """
        logger.info(
            f"Beginning automated model selection for {wide_df.shape[1]} series "
            f"(horizon={self.config.horizon}, metric={self.config.metric})"
        )
        records = []
        for col in wide_df.columns:
            s = wide_df[col]
            res = self.evaluate_series(s, series_id=str(col))

            naive_score = res.all_scores.get(
                "Seasonal_Naive", res.all_scores.get("Naive_Last", res.champion_score)
            )
            improvement = (
                round(
                    ((naive_score - res.champion_score) / naive_score) * 100.0, 2
                )
                if naive_score > 0
                else 0.0
            )

            records.append({
                "Series ID": res.series_id,
                "Regime": res.regime,
                "ADI": res.adi,
                "CV2": res.cv2,
                "Champion Model": res.champion_model,
                f"Champion {self.config.metric.upper()} (%)": res.champion_score,
                "Runner Up Model": res.runner_up_model,
                f"Runner Up {self.config.metric.upper()} (%)": res.runner_up_score,
                "Selection Margin": res.selection_margin,
                "Improvement vs Naive (%)": improvement,
            })

        leaderboard_df = pd.DataFrame(records)
        return self.routing_map, leaderboard_df

    def save_champion_routing(
        self,
        filepath: Union[str, Path],
    ) -> None:
        """Saves champion routing map and metadata to JSON for downstream pipelines."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "metric": self.config.metric,
                "horizon": self.config.horizon,
                "n_series": len(self.routing_map),
            },
            "routing_map": self.routing_map,
            "results": {
                k: v.to_dict() for k, v in self.selection_results.items()
            },
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Saved champion model routing map to {p}")

