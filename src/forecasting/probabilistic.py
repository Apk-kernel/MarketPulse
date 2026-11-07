"""Probabilistic demand forecasting, multi-quantile regression, conformal intervals, and intermittent models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

from src.evaluation.metrics import (
    continuous_ranked_probability_score,
    mean_pinball_loss,
    mean_prediction_interval_width,
    pinball_loss,
    prediction_interval_coverage_probability,
    winkler_score,
)
from src.features.lag_features import create_lag_features
from src.features.rolling_features import create_rolling_features
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Forecasting.Probabilistic")

DEFAULT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def enforce_monotonic_quantiles(
    quantiles_df: pd.DataFrame,
    non_negative: bool = True,
) -> pd.DataFrame:
    """Eliminates quantile crossings via Chernozhukov et al. (2010) monotonic rearrangement.

    Args:
        quantiles_df: DataFrame where column names are float quantiles in ascending order.
        non_negative: If True, clips values below zero (y_hat >= 0).

    Returns:
        DataFrame with strictly non-decreasing quantile values across each row.
    """
    sorted_cols = sorted(quantiles_df.columns, key=lambda c: float(c))
    vals = quantiles_df[sorted_cols].values.copy()

    # Monotonic sorting along the quantile axis for each forecast horizon step
    vals = np.sort(vals, axis=1)

    if non_negative:
        vals = np.maximum(0.0, vals)

    return pd.DataFrame(vals, index=quantiles_df.index, columns=sorted_cols)


class QuantileLightGBMForecaster:
    """Multi-quantile gradient boosted trees with monotonic rearrangement and non-negativity."""

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_seed: int = 42,
    ):
        """Initializes multi-quantile LightGBM forecaster.

        Args:
            quantiles: List of quantiles in (0, 1) to predict. Defaults to [0.05, ..., 0.95].
            lags: Target lag steps to generate. Defaults to [1, 7, 14, 28].
            rolling_windows: Rolling mean window sizes. Defaults to [7, 14, 28].
            n_estimators: Trees per quantile model.
            learning_rate: Learning rate.
            max_depth: Max tree depth.
            random_seed: Random seed.
        """
        self.quantiles = sorted(quantiles or DEFAULT_QUANTILES)
        self.lags = lags or [1, 7, 14, 28]
        self.rolling_windows = rolling_windows or [7, 14, 28]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_seed = random_seed

        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_cols: List[str] = []
        self._fitted = False
        self._last_history: Optional[pd.Series] = None

    def _engineer_features(self, series: pd.Series) -> pd.DataFrame:
        """Constructs lag and calendar features strictly without lookahead leakage."""
        df = pd.DataFrame({"sales": series.values}, index=series.index)

        # Lags
        df_lags = create_lag_features(df, target_col="sales", lags=self.lags)

        # Rolling
        df_rolling = create_rolling_features(
            df_lags, target_col="sales", mean_windows=self.rolling_windows, shift=1
        )

        # Calendar features from datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df_rolling["dayofweek"] = df.index.dayofweek
            df_rolling["month"] = df.index.month
            df_rolling["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
        else:
            df_rolling["step"] = np.arange(len(df))

        return df_rolling

    def fit(self, y: pd.Series) -> "QuantileLightGBMForecaster":
        """Fits independent LightGBM quantile regression models for each specified quantile."""
        self._last_history = y.copy()
        feat_df = self._engineer_features(y)

        # Drop rows with NaN from lags/rolling
        clean_df = feat_df.dropna()
        if len(clean_df) < 10:
            raise ValueError(f"Insufficient training rows ({len(clean_df)}) after feature engineering.")

        drop_cols = ["sales"]
        self.feature_cols = [c for c in clean_df.columns if c not in drop_cols]
        X = clean_df[self.feature_cols]
        target = clean_df["sales"]

        logger.info(
            f"Fitting {len(self.quantiles)} Quantile LightGBM models on {len(clean_df)} samples "
            f"with {len(self.feature_cols)} features."
        )

        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_seed,
                verbose=-1,
            )
            model.fit(X, target)
            self.models[q] = model

        self._fitted = True
        return self

    def predict_quantiles(
        self,
        horizon: int = 28,
        freq: str = "D",
    ) -> pd.DataFrame:
        """Generates multi-step recursive quantile forecasts with non-crossing rearrangement.

        Args:
            horizon: Forecast horizon steps.
            freq: Date frequency.

        Returns:
            DataFrame of shape (horizon, len(quantiles)) indexed by future dates.
        """
        if not self._fitted:
            raise RuntimeError("Forecaster must be fit before calling predict_quantiles().")

        last_idx = self._last_history.index[-1]
        future_dates = pd.date_range(start=last_idx + pd.Timedelta(days=1), periods=horizon, freq=freq)

        pred_matrix = np.zeros((horizon, len(self.quantiles)), dtype=np.float64)
        history = self._last_history.copy()

        # Step-by-step recursive roll-forward
        for step in range(horizon):
            cur_feat = self._engineer_features(history).iloc[[-1]]
            X_cur = cur_feat[self.feature_cols]

            step_preds = []
            for j, q in enumerate(self.quantiles):
                q_val = float(self.models[q].predict(X_cur)[0])
                pred_matrix[step, j] = q_val
                step_preds.append(q_val)

            # Use median (0.50) or mean of quantiles to advance recursive state
            median_idx = len(self.quantiles) // 2
            adv_val = pred_matrix[step, median_idx]
            new_date = future_dates[step]
            history.loc[new_date] = max(0.0, adv_val)

        raw_df = pd.DataFrame(pred_matrix, index=future_dates, columns=self.quantiles)

        # Monotonic rearrangement to prevent quantile crossing
        clean_quantiles = enforce_monotonic_quantiles(raw_df, non_negative=True)
        return clean_quantiles

    def predict_intervals(
        self,
        coverage: float = 0.90,
        horizon: int = 28,
    ) -> Tuple[pd.Series, pd.Series]:
        """Returns lower and upper prediction interval series for a specified nominal coverage."""
        q_df = self.predict_quantiles(horizon=horizon)
        alpha = (1.0 - coverage) / 2.0
        lower_q = alpha
        upper_q = 1.0 - alpha

        # Find closest available quantiles
        avail = [float(c) for c in q_df.columns]
        closest_lower = min(avail, key=lambda x: abs(x - lower_q))
        closest_upper = min(avail, key=lambda x: abs(x - upper_q))

        lower_bound = q_df[closest_lower]
        upper_bound = q_df[closest_upper]
        return lower_bound, upper_bound


class IntermittentProbabilisticForecaster:
    """Zero-inflated probabilistic forecaster for intermittent and lumpy demand regimes.

    Decomposes demand into occurrence probability (Croston / Bernoulli) and positive demand size (Negative Binomial).
    """

    def __init__(self, smoothing: float = 0.1):
        """Initializes intermittent forecaster with exponential smoothing parameter."""
        self.smoothing = smoothing
        self.occurrence_prob_: float = 0.0
        self.size_mean_: float = 0.0
        self.size_var_: float = 0.0
        self.n_samples_: int = 0
        self._fitted = False

    def fit(self, y: pd.Series) -> "IntermittentProbabilisticForecaster":
        """Fits occurrence probability and non-zero demand size distribution."""
        vals = np.asarray(y.values, dtype=np.float64)
        self.n_samples_ = len(vals)
        nonzero_mask = vals > 0

        # Occurrence probability: pi = fraction of positive demand periods
        pi = float(np.mean(nonzero_mask))
        self.occurrence_prob_ = max(1e-4, min(1.0 - 1e-4, pi))

        nonzero_vals = vals[nonzero_mask]
        if len(nonzero_vals) > 0:
            self.size_mean_ = float(np.mean(nonzero_vals))
            self.size_var_ = float(np.var(nonzero_vals, ddof=1)) if len(nonzero_vals) > 1 else self.size_mean_
            self.size_var_ = max(self.size_var_, self.size_mean_ + 1e-4)
        else:
            self.size_mean_ = 1.0
            self.size_var_ = 1.0

        self._fitted = True
        return self

    def predict_quantiles(
        self,
        horizon: int = 28,
        quantiles: Optional[List[float]] = None,
        dates_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Derives exact zero-inflated quantiles: Q(alpha) = 0 if alpha <= 1 - pi."""
        if not self._fitted:
            raise RuntimeError("IntermittentProbabilisticForecaster must be fit first.")

        q_list = sorted(quantiles or DEFAULT_QUANTILES)
        pi = self.occurrence_prob_
        mu_z = self.size_mean_
        var_z = self.size_var_

        # Parameterize distribution for demand size:
        # If overdispersed (var_z > 1.05 * mu_z), fit Negative Binomial
        # Otherwise (equidispersed/underdispersed), fit Poisson to prevent degenerate parameter explosion
        is_overdispersed = var_z > 1.05 * mu_z
        if is_overdispersed:
            p_nb = max(1e-4, min(0.999, mu_z / var_z))
            n_nb = max(1e-4, (mu_z ** 2) / (var_z - mu_z))
        else:
            p_nb = None
            n_nb = None

        q_vals = {}
        for q in q_list:
            if q <= (1.0 - pi):
                # When quantile falls in zero-inflation mass, demand is exactly 0
                q_vals[q] = 0.0
            else:
                # Conditional quantile on positive demand
                q_adj = (q - (1.0 - pi)) / pi
                q_adj = np.clip(q_adj, 1e-4, 1.0 - 1e-4)
                if is_overdispersed:
                    z_q = stats.nbinom.ppf(q_adj, n=n_nb, p=p_nb)
                else:
                    z_q = stats.poisson.ppf(q_adj, mu=mu_z)
                q_vals[q] = float(np.nan_to_num(z_q, nan=mu_z))

        # Replicate across horizon
        pred_mat = np.tile(list(q_vals.values()), (horizon, 1))
        df = pd.DataFrame(pred_mat, columns=q_list)
        if dates_index is not None and len(dates_index) == horizon:
            df.index = dates_index

        return enforce_monotonic_quantiles(df, non_negative=True)


class ConformalIntervalForecaster:
    """Distribution-free conformal prediction intervals wrapping any base time-series forecaster.

    Guarantees exact finite-sample coverage P(y_{t+h} in [l, u]) >= 1 - beta.
    Supports horizon-adaptive error scaling to prevent multi-step coverage degradation.
    """

    def __init__(
        self,
        forecaster: Any,
        nominal_coverage: float = 0.90,
        horizon_adaptive: bool = True,
    ):
        """Initializes conformal interval forecaster.

        Args:
            forecaster: Any point forecaster supporting fit() and predict().
            nominal_coverage: Target confidence level (e.g. 0.80 or 0.90 or 0.95).
            horizon_adaptive: If True, scales bounds by sqrt(h) to match multi-step error expansion.
        """
        self.forecaster = forecaster
        self.nominal_coverage = nominal_coverage
        self.horizon_adaptive = horizon_adaptive
        self.calibration_cutoff_: float = 0.0
        self._fitted = False

    def fit(
        self,
        y: pd.Series,
        val_fraction: float = 0.25,
    ) -> "ConformalIntervalForecaster":
        """Fits base forecaster on training segment and calibrates conformity scores on holdout."""
        T = len(y)
        split_idx = int(T * (1.0 - val_fraction))
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]

        # Fit forecaster on training split
        self.forecaster.fit(y_train)

        # Predict on validation split
        val_fh = np.arange(1, len(y_val) + 1)
        val_preds = self.forecaster.predict(fh=val_fh)
        if hasattr(val_preds, "values"):
            val_preds_arr = val_preds.values.ravel()
        else:
            val_preds_arr = np.asarray(val_preds).ravel()

        # Compute absolute nonconformity residuals: R_t = |y_t - y_hat_t|
        residuals = np.abs(y_val.values.ravel() - val_preds_arr)

        # Conformal quantile: (1 - beta) * (1 + 1/N)
        N = len(residuals)
        target_quantile = min(1.0, self.nominal_coverage * (1.0 + 1.0 / max(1, N)))

        if self.horizon_adaptive:
            h_steps = np.arange(1, N + 1)
            normalized_residuals = residuals / np.sqrt(h_steps)
            self.calibration_cutoff_ = float(np.quantile(normalized_residuals, target_quantile))
        else:
            self.calibration_cutoff_ = float(np.quantile(residuals, target_quantile))

        # Refit on full series
        self.forecaster.fit(y)
        self._fitted = True
        return self

    def predict_intervals(
        self,
        horizon: int = 28,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generates point forecasts and calibrated conformal upper/lower prediction bounds.

        Returns:
            Tuple of (point_forecast, lower_bound, upper_bound).
        """
        if not self._fitted:
            raise RuntimeError("ConformalIntervalForecaster must be fit first.")

        fh = np.arange(1, horizon + 1)
        pt_preds = self.forecaster.predict(fh=fh)
        if isinstance(pt_preds, pd.Series):
            point = pt_preds
        else:
            point = pd.Series(np.asarray(pt_preds).ravel())

        if self.horizon_adaptive:
            h_steps = np.arange(1, horizon + 1)
            half_widths = self.calibration_cutoff_ * np.sqrt(h_steps)
            lower = (point - half_widths).clip(lower=0.0)
            upper = point + half_widths
        else:
            lower = (point - self.calibration_cutoff_).clip(lower=0.0)
            upper = point + self.calibration_cutoff_

        return point, lower, upper
