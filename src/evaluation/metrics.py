"""Statistical evaluation metrics for demand forecasting.

Implements MAE, RMSE, sMAPE, WAPE, and MASE with strict zero-division protection.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Error (MAE)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error (RMSE)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Calculates symmetric Mean Absolute Percentage Error (sMAPE) in percentage [0, 200].

    Formula:
        sMAPE = (200 / H) * sum(|y_t - y_hat_t| / (|y_t| + |y_hat_t| + eps))
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_t) + np.abs(y_p) + epsilon
    smape = 200.0 * np.mean(np.abs(y_t - y_p) / denominator)
    return float(smape)


def weighted_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Calculates Weighted Absolute Percentage Error (WAPE / MAD/Mean ratio).

    Formula:
        WAPE = sum(|y_t - y_hat_t|) / (sum(y_t) + eps) * 100
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    total_actual = np.sum(np.abs(y_t))
    if total_actual < epsilon:
        return 0.0 if np.sum(np.abs(y_p)) < epsilon else 100.0
    wape = (np.sum(np.abs(y_t - y_p)) / (total_actual + epsilon)) * 100.0
    return float(wape)


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    sp: int = 1,
    epsilon: float = 1e-8,
) -> float:
    """Calculates Mean Absolute Scaled Error (MASE) relative to in-sample naive forecast.

    Formula:
        scale = mean(|y_train[t] - y_train[t-sp]|)
        MASE = MAE(y_true, y_pred) / (scale + eps)
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)

    if len(y_tr) <= sp:
        return 1.0

    in_sample_mae = np.mean(np.abs(y_tr[sp:] - y_tr[:-sp]))
    if in_sample_mae < epsilon:
        return 1.0

    mae = np.mean(np.abs(y_t - y_p))
    return float(mae / (in_sample_mae + epsilon))


def root_mean_squared_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    after_first_nonzero: bool = True,
    epsilon: float = 1e-8,
) -> float:
    """Calculates M5 Root Mean Squared Scaled Error (RMSSE) for a single series.

    Formula:
        Numerator = mean((y_true - y_pred)^2)
        Denominator = (1 / (n - 1)) * sum_{t=2}^n (y_train[t] - y_train[t-1])^2
        RMSSE = sqrt(Numerator / (Denominator + epsilon))

    Args:
        y_true: Ground truth test demand.
        y_pred: Forecasted demand.
        y_train: In-sample historical demand.
        after_first_nonzero: If True, computes scale only after the first positive sale
                            (M5 competition standard for unlisted/launch periods).
        epsilon: Zero-division guard.

    Returns:
        RMSSE scalar.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ytr = np.asarray(y_train, dtype=float)

    if after_first_nonzero:
        nonzero_idx = np.where(ytr > 0)[0]
        if len(nonzero_idx) > 0:
            ytr = ytr[nonzero_idx[0]:]

    if len(ytr) < 2:
        return 1.0

    # Scale: in-sample naive diff squared mean
    scale_sq = np.mean((ytr[1:] - ytr[:-1]) ** 2)
    if scale_sq < epsilon:
        scale_sq = epsilon

    mse = np.mean((yt - yp) ** 2)
    return float(np.sqrt(mse / scale_sq))


def weighted_root_mean_squared_scaled_error(
    rmsse_values: Union[List[float], np.ndarray],
    weights: Union[List[float], np.ndarray],
    normalize_weights: bool = True,
) -> float:
    """Calculates Weighted Root Mean Squared Scaled Error (WRMSSE) across multiple series.

    Formula:
        WRMSSE = sum_i (w_i * RMSSE_i)

    Args:
        rmsse_values: Array/list of RMSSE values for each series.
        weights: Dollar sales volume or importance weights for each series.
        normalize_weights: If True, normalizes weights to sum to 1.0.

    Returns:
        WRMSSE scalar.
    """
    r_arr = np.asarray(rmsse_values, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    if len(r_arr) == 0:
        return 0.0

    if normalize_weights:
        total_w = np.sum(w_arr)
        if total_w > 0:
            w_arr = w_arr / total_w
        else:
            w_arr = np.ones_like(w_arr) / len(w_arr)

    return float(np.sum(w_arr * r_arr))


def mean_absolute_percentage_error_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_threshold: float = 1e-4,
) -> float:
    """Calculates MAPE strictly on non-zero actuals to prevent mathematical division by zero.

    Args:
        y_true: Ground truth actual values.
        y_pred: Predicted values.
        zero_threshold: Cutoff below which actuals are excluded from percentage error.

    Returns:
        MAPE percentage [0, 100+], or NaN if all actuals are zero.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    non_zero_mask = np.abs(yt) > zero_threshold

    if not np.any(non_zero_mask):
        return np.nan

    mape = np.mean(np.abs((yt[non_zero_mask] - yp[non_zero_mask]) / yt[non_zero_mask])) * 100.0
    return float(mape)


def calculate_forecast_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_train: Optional[Union[pd.Series, np.ndarray]] = None,
    sp: int = 7,
) -> Dict[str, float]:
    """Calculates comprehensive forecast evaluation metrics dictionary.

    Args:
        y_true: Actual demand values over forecast horizon.
        y_pred: Forecasted demand values.
        y_train: Optional in-sample historical observations for MASE scaling.
        sp: Seasonal periodicity (e.g. 7 for weekly seasonality).

    Returns:
        Dictionary containing MAE, RMSE, sMAPE, WAPE, and MASE.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    metrics = {
        "MAE": round(mean_absolute_error(yt, yp), 4),
        "RMSE": round(root_mean_squared_error(yt, yp), 4),
        "sMAPE": round(symmetric_mean_absolute_percentage_error(yt, yp), 2),
        "WAPE": round(weighted_absolute_percentage_error(yt, yp), 2),
    }

    if y_train is not None and len(y_train) > sp:
        y_train_arr = np.asarray(y_train, dtype=float)
        metrics["MASE"] = round(mean_absolute_scaled_error(yt, yp, y_train_arr, sp=sp), 4)
        metrics["RMSSE"] = round(root_mean_squared_scaled_error(yt, yp, y_train_arr), 4)

    return metrics


# ============================================================================
# Probabilistic Forecasting Evaluation Metrics
# ============================================================================

def pinball_loss(
    y_true: Union[pd.Series, np.ndarray],
    y_pred_quantile: Union[pd.Series, np.ndarray],
    alpha: float,
) -> float:
    """Calculates Pinball Loss (Quantile Loss) for a specified target quantile alpha in (0, 1).

    Formula:
        L_alpha(y, q) = max(alpha * (y - q), (1 - alpha) * (q - y))
        Pinball_Loss = mean(L_alpha(y_t, q_t))

    Args:
        y_true: Actual demand values.
        y_pred_quantile: Predicted quantile values.
        alpha: Target quantile in (0, 1).

    Returns:
        Pinball loss scalar.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    yt = np.asarray(y_true, dtype=np.float64)
    yq = np.asarray(y_pred_quantile, dtype=np.float64)
    diff = yt - yq
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def mean_pinball_loss(
    y_true: Union[pd.Series, np.ndarray],
    quantiles_df: pd.DataFrame,
) -> float:
    """Calculates average pinball loss across all evaluated quantiles.

    Args:
        y_true: Actual ground truth observations.
        quantiles_df: DataFrame where column names are float quantiles (e.g. 0.05, 0.50, 0.95).

    Returns:
        Average pinball loss scalar.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    losses = []
    for col in quantiles_df.columns:
        alpha = float(col)
        yq = quantiles_df[col].values
        losses.append(pinball_loss(yt, yq, alpha=alpha))
    return float(np.mean(losses))


def prediction_interval_coverage_probability(
    y_true: Union[pd.Series, np.ndarray],
    lower_bound: Union[pd.Series, np.ndarray],
    upper_bound: Union[pd.Series, np.ndarray],
) -> float:
    """Calculates Prediction Interval Coverage Probability (PICP) in percent [0, 100].

    Formula:
        PICP = (1 / H) * sum_{t=1}^H I(lower_t <= y_t <= upper_t) * 100%

    Args:
        y_true: Actual demand observations.
        lower_bound: Lower prediction interval boundary.
        upper_bound: Upper prediction interval boundary.

    Returns:
        Percentage of actuals falling within the prediction interval.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    l_b = np.asarray(lower_bound, dtype=np.float64)
    u_b = np.asarray(upper_bound, dtype=np.float64)

    covered = (yt >= l_b) & (yt <= u_b)
    return float(np.mean(covered) * 100.0)


def mean_prediction_interval_width(
    lower_bound: Union[pd.Series, np.ndarray],
    upper_bound: Union[pd.Series, np.ndarray],
) -> float:
    """Calculates Mean Prediction Interval Width (MPIW / sharpness metric).

    Formula:
        MPIW = (1 / H) * sum_{t=1}^H (upper_t - lower_t)

    Args:
        lower_bound: Lower prediction interval boundary.
        upper_bound: Upper prediction interval boundary.

    Returns:
        Average interval width scalar.
    """
    l_b = np.asarray(lower_bound, dtype=np.float64)
    u_b = np.asarray(upper_bound, dtype=np.float64)
    return float(np.mean(np.maximum(0.0, u_b - l_b)))


def winkler_score(
    y_true: Union[pd.Series, np.ndarray],
    lower_bound: Union[pd.Series, np.ndarray],
    upper_bound: Union[pd.Series, np.ndarray],
    nominal_coverage: float = 0.95,
) -> float:
    """Calculates Winkler Score (Interval Score) combining sharpness and penalties for misses.

    Formula (for nominal coverage 1 - alpha):
        Score_t = (u_t - l_t)
                + (2 / alpha) * (l_t - y_t) * I(y_t < l_t)
                + (2 / alpha) * (y_t - u_t) * I(y_t > u_t)

    Args:
        y_true: Actual demand observations.
        lower_bound: Lower prediction interval boundary.
        upper_bound: Upper prediction interval boundary.
        nominal_coverage: Expected coverage (e.g. 0.95 or 0.80).

    Returns:
        Average Winkler score (lower is better).
    """
    alpha = 1.0 - nominal_coverage
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"nominal_coverage must be in (0, 1), got {nominal_coverage}")

    yt = np.asarray(y_true, dtype=np.float64)
    l_b = np.asarray(lower_bound, dtype=np.float64)
    u_b = np.asarray(upper_bound, dtype=np.float64)

    width = np.maximum(0.0, u_b - l_b)
    penalty_lower = (2.0 / alpha) * np.maximum(0.0, l_b - yt)
    penalty_upper = (2.0 / alpha) * np.maximum(0.0, yt - u_b)

    total_score = width + penalty_lower + penalty_upper
    return float(np.mean(total_score))


def continuous_ranked_probability_score(
    y_true: Union[pd.Series, np.ndarray],
    quantiles_df: pd.DataFrame,
) -> float:
    """Calculates Continuous Ranked Probability Score (CRPS) approximated over empirical quantiles.

    Following Gneiting & Raftery (2007):
        CRPS(y, F) approx = (2 / K) * sum_{k=1}^K L_{alpha_k}(y, q_{alpha_k})
    where L_alpha is the pinball loss.

    Args:
        y_true: Actual demand observations.
        quantiles_df: DataFrame where column names are float quantiles in ascending order.

    Returns:
        CRPS approximation scalar.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    alphas = [float(c) for c in quantiles_df.columns]
    K = len(alphas)
    if K == 0:
        return 0.0

    total_loss = 0.0
    for alpha in alphas:
        yq = quantiles_df[alpha].values
        total_loss += pinball_loss(yt, yq, alpha=alpha)

    crps = (2.0 / K) * total_loss
    return float(crps)


