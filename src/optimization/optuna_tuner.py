"""Optuna-based Bayesian hyperparameter optimization engine for demand forecasting.

Provides:
1. LightGBM hyperparameter optimization over temporal rolling splits.
2. Median/ASHA pruning of underperforming hyperparameter candidates.
3. Feature and lag window length joint optimization.
4. Parameter importance analysis and study visualization.
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, NopPruner
from lightgbm import LGBMRegressor
from sktime.forecasting.compose import make_reduction

from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.forecasting.base import DemandForecasterWrapper
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Optimization.OptunaTuner")

# Silence noisy optuna and scikit-learn warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptunaTuningConfig:
    """Configuration container for Optuna Bayesian hyperparameter search."""

    n_trials: int = 25
    timeout_sec: Optional[int] = None
    metric: str = "wape"  # "wape", "rmse", "mae"
    horizon: int = 14
    n_splits: int = 2
    pruning: bool = True
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LightGBMHyperparameterTuner:
    """Orchestrates Bayesian hyperparameter optimization for LightGBM demand forecasters."""

    def __init__(self, config: Optional[OptunaTuningConfig] = None):
        """Initializes tuner with search configuration.

        Args:
            config: Optimization settings (trials, metric, pruning, seed).
        """
        self.config = config or OptunaTuningConfig()
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None

    def sample_hyperparameters(self, trial: optuna.Trial) -> Tuple[Dict[str, Any], int]:
        """Samples LightGBM hyperparameters and tabular window length.

        Args:
            trial: Optuna trial object.

        Returns:
            Tuple of (lgbm_params_dict, window_length_int).
        """
        lgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 100, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "verbose": -1,
            "random_state": self.config.random_seed,
            "n_jobs": 1,
        }
        window_length = trial.suggest_categorical("window_length", [7, 14, 21])
        return lgb_params, window_length

    def evaluate_split(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        reg_params: Dict[str, Any],
        window_length: int,
    ) -> float:
        """Evaluates model performance on a single temporal split.

        Args:
            y_train: Historical training series.
            y_test: Out-of-time evaluation series.
            reg_params: LightGBM hyperparameters.
            window_length: Reduction lag window length.

        Returns:
            Computed validation metric (WAPE, RMSE, or MAE).
        """
        # Ensure window_length does not exceed training sample size
        effective_window = min(window_length, len(y_train) - 2)
        if effective_window < 2:
            return 999.0

        reg = LGBMRegressor(**reg_params)
        forecaster = make_reduction(
            estimator=reg,
            window_length=effective_window,
            strategy="recursive",
            scitype="tabular-regressor",
        )

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
        else:  # Default to WAPE
            return float(weighted_absolute_percentage_error(act_vals, pred_vals))

    def objective(self, trial: optuna.Trial, y: pd.Series) -> float:
        """Objective function evaluating hyperparameter candidates across temporal folds."""
        lgb_params, window_length = self.sample_hyperparameters(trial)
        T = len(y)
        fh = self.config.horizon
        n_splits = self.config.n_splits

        # Ensure enough observations for n_splits * fh validation windows
        required_len = (n_splits + 1) * fh
        if T < required_len:
            # Fall back to single split
            n_splits = 1

        fold_errors = []
        for fold in range(n_splits):
            test_start = T - (n_splits - fold) * fh
            test_end = test_start + fh

            y_tr = y.iloc[:test_start]
            y_te = y.iloc[test_start:test_end]

            err = self.evaluate_split(y_tr, y_te, lgb_params, window_length)
            fold_errors.append(err)

            # Pruning check
            if self.config.pruning:
                trial.report(err, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return float(np.mean(fold_errors))

    def optimize(
        self,
        y: pd.Series,
        study_name: str = "lgbm_demand_optimization",
    ) -> optuna.Study:
        """Executes Bayesian hyperparameter optimization study.

        Args:
            y: Historical demand series (DatetimeIndex with daily frequency).
            study_name: Identifier for Optuna study.

        Returns:
            Completed Optuna study instance.
        """
        logger.info(
            f"Starting Optuna hyperparameter optimization: trials={self.config.n_trials}, "
            f"metric={self.config.metric}, horizon={self.config.horizon}"
        )

        sampler = TPESampler(seed=self.config.random_seed)
        pruner = (
            MedianPruner(n_startup_trials=5, n_warmup_steps=1)
            if self.config.pruning
            else NopPruner()
        )

        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        self.study.optimize(
            lambda trial: self.objective(trial, y),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_sec,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(
            f"Optimization finished: Best Trial #{self.study.best_trial.number} "
            f"with {self.config.metric.upper()}={self.best_value:.3f}"
        )
        return self.study

    def get_best_estimator(self) -> DemandForecasterWrapper:
        """Builds optimal DemandForecasterWrapper parameterized with best trial values."""
        if self.study is None or self.best_params is None:
            raise RuntimeError("Must call optimize() before building best estimator.")

        p = self.best_params.copy()
        window_length = p.pop("window_length", 14)

        reg_params = {
            "n_estimators": p.get("n_estimators", 50),
            "learning_rate": p.get("learning_rate", 0.05),
            "num_leaves": p.get("num_leaves", 31),
            "max_depth": p.get("max_depth", 6),
            "min_child_samples": p.get("min_child_samples", 20),
            "subsample": p.get("subsample", 0.8),
            "subsample_freq": 1,
            "colsample_bytree": p.get("colsample_bytree", 0.8),
            "reg_alpha": p.get("reg_alpha", 0.1),
            "reg_lambda": p.get("reg_lambda", 0.1),
            "verbose": -1,
            "random_state": self.config.random_seed,
            "n_jobs": 1,
        }

        reg = LGBMRegressor(**reg_params)
        forecaster = make_reduction(
            estimator=reg,
            window_length=window_length,
            strategy="recursive",
            scitype="tabular-regressor",
        )

        return DemandForecasterWrapper(
            name=f"Tuned_LightGBM_w{window_length}",
            estimator=forecaster,
            clip_non_negative=True,
            params={"lgbm_params": reg_params, "window_length": window_length},
        )

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Returns structured DataFrame of all completed trials."""
        if self.study is None:
            raise RuntimeError("Study not executed yet.")

        df_trials = self.study.trials_dataframe()
        return df_trials.sort_values("value").reset_index(drop=True)

    def get_param_importance(self) -> Dict[str, float]:
        """Calculates fANOVA / MDI hyperparameter importance scores."""
        if self.study is None:
            raise RuntimeError("Study not executed yet.")

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return {k: round(float(v), 4) for k, v in importances.items()}
        except Exception as e:
            logger.warning(f"Could not compute parameter importances: {e}")
            return {}

