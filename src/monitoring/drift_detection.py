"""Data and Covariate Shift Detection Engine for MarketPulse.

Implements statistical hypothesis testing and distribution divergence metrics:
1. Population Stability Index (PSI) with decile discretization.
2. Two-Sample Kolmogorov-Smirnov (KS) test for empirical CDF divergence.
3. Wasserstein-1 Distance (Earth Mover's Distance) with scale normalization.
4. Categorical Jensen-Shannon Divergence and Chi-Square frequency shift.
5. Target demand volume, sparsity, and Syntetos-Boylan demand regime shifts.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from src.features.demand_features import calculate_demand_profile
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Monitoring.DriftDetection")


@dataclass
class FeatureDriftResult:
    """Detailed drift evaluation for a single feature."""

    feature_name: str
    feature_type: str  # 'numerical' or 'categorical'
    psi: float
    psi_status: str  # 'STABLE', 'MODERATE_DRIFT', 'SEVERE_DRIFT'
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    ks_drift_detected: Optional[bool] = None
    wasserstein_dist: Optional[float] = None
    relative_wasserstein: Optional[float] = None
    js_divergence: Optional[float] = None
    sample_size_ref: int = 0
    sample_size_curr: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Tuple[float, pd.DataFrame]:
    """Computes the Population Stability Index (PSI) between reference and current samples.

    PSI Formula:
        PSI = sum_b (Actual_b - Expected_b) * ln( (Actual_b + eps) / (Expected_b + eps) )

    Thresholds:
        PSI < 0.10: Stable distribution (No action needed)
        0.10 <= PSI < 0.25: Moderate shift (Monitor closely)
        PSI >= 0.25: Significant shift (Actionable covariate drift)

    Args:
        reference: Historical baseline training values.
        current: Operational live evaluation values.
        num_bins: Target number of quantiles for binning (default: 10).
        epsilon: Small smoothing constant to avoid log(0) and division by zero.

    Returns:
        Tuple of (total_psi_score, bin_breakdown_dataframe).
    """
    ref_clean = np.asarray(reference, dtype=float)
    curr_clean = np.asarray(current, dtype=float)

    ref_clean = ref_clean[np.isfinite(ref_clean)]
    curr_clean = curr_clean[np.isfinite(curr_clean)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0, pd.DataFrame()

    # Determine bin edges from reference sample quantiles
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles * 100.0)
    # Ensure strictly increasing unique bin edges
    unique_edges = np.unique(bin_edges)

    if len(unique_edges) < 2:
        all_vals = np.concatenate([ref_clean, curr_clean])
        all_unique = np.unique(all_vals)
        if len(all_unique) < 2:
            return 0.0, pd.DataFrame()
        # Construct bin edges between unique values across both samples
        mids = (all_unique[:-1] + all_unique[1:]) / 2.0
        unique_edges = np.concatenate([[-np.inf], mids, [np.inf]])
    else:
        # Adjust outermost edges to encompass entire domain
        unique_edges[0] = -np.inf
        unique_edges[-1] = np.inf

    # Discretize both samples into identical bins
    ref_counts, _ = np.histogram(ref_clean, bins=unique_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=unique_edges)

    # Convert to probability distributions
    ref_props = ref_counts / len(ref_clean)
    curr_props = curr_counts / len(curr_clean)

    # Compute per-bin PSI components
    psi_components = (curr_props - ref_props) * np.log((curr_props + epsilon) / (ref_props + epsilon))
    total_psi = float(np.sum(psi_components))

    # Construct interpretable bin breakdown
    bin_labels = [
        f"[{round(unique_edges[i], 3)}, {round(unique_edges[i+1], 3)})"
        for i in range(len(unique_edges) - 1)
    ]
    breakdown_df = pd.DataFrame({
        "bin_range": bin_labels,
        "ref_count": ref_counts,
        "curr_count": curr_counts,
        "ref_pct": np.round(ref_props * 100.0, 2),
        "curr_pct": np.round(curr_props * 100.0, 2),
        "psi_component": np.round(psi_components, 5),
    })

    return round(total_psi, 4), breakdown_df


def classify_psi(psi_value: float) -> str:
    """Classifies PSI score into industry standard operational risk categories."""
    if psi_value < 0.10:
        return "STABLE"
    elif psi_value < 0.25:
        return "MODERATE_DRIFT"
    else:
        return "SEVERE_DRIFT"


def kolmogorov_smirnov_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Performs two-sample Kolmogorov-Smirnov test for continuous feature distribution divergence.

    Null hypothesis H0: Reference and Current samples are drawn from the same continuous distribution.

    Args:
        reference: Reference sample values.
        current: Current operational sample values.
        alpha: Statistical significance threshold (default: 0.05).

    Returns:
        Dictionary with KS statistic, p-value, and drift detection boolean.
    """
    ref_clean = reference[np.isfinite(reference)]
    curr_clean = current[np.isfinite(current)]

    if len(ref_clean) < 5 or len(curr_clean) < 5:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    res = stats.ks_2samp(ref_clean, curr_clean)
    return {
        "statistic": round(float(res.statistic), 4),
        "p_value": round(float(res.pvalue), 6),
        "drift_detected": bool(res.pvalue < alpha),
    }


def wasserstein_distance_test(
    reference: np.ndarray,
    current: np.ndarray,
) -> Dict[str, float]:
    """Computes Wasserstein-1 distance (Earth Mover's Distance) and scale-normalized distance.

    Args:
        reference: Reference sample values.
        current: Current operational sample values.

    Returns:
        Dictionary containing absolute and relative Wasserstein distances.
    """
    ref_clean = reference[np.isfinite(reference)]
    curr_clean = current[np.isfinite(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return {"wasserstein_dist": 0.0, "relative_wasserstein": 0.0}

    w_dist = float(stats.wasserstein_distance(ref_clean, curr_clean))
    ref_std = float(np.std(ref_clean))
    rel_w = w_dist / (ref_std + 1e-6)

    return {
        "wasserstein_dist": round(w_dist, 4),
        "relative_wasserstein": round(rel_w, 4),
    }


def categorical_divergence_test(
    reference: np.ndarray,
    current: np.ndarray,
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Computes PSI and Jensen-Shannon Divergence for discrete / categorical distributions.

    Args:
        reference: Reference categorical array.
        current: Current categorical array.
        epsilon: Smoothing constant.

    Returns:
        Dictionary with categorical PSI, JS divergence, and status.
    """
    all_categories = np.unique(np.concatenate([reference, current]))
    ref_counts = pd.Series(reference).value_counts().reindex(all_categories, fill_value=0)
    curr_counts = pd.Series(current).value_counts().reindex(all_categories, fill_value=0)

    p = (ref_counts.values / len(reference)).astype(float)
    q = (curr_counts.values / len(current)).astype(float)

    # PSI calculation
    psi_comp = (q - p) * np.log((q + epsilon) / (p + epsilon))
    psi = float(np.sum(psi_comp))

    # Jensen-Shannon Divergence [0, 1]
    js_div = float(jensenshannon(p, q, base=2))

    return {
        "psi": round(psi, 4),
        "psi_status": classify_psi(psi),
        "js_divergence": round(js_div, 4),
    }


class DataDriftDetector:
    """Enterprise-grade data drift and covariate shift monitoring engine."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        alpha: float = 0.05,
    ):
        """Initializes DataDriftDetector with a baseline reference dataset.

        Args:
            reference_df: Historical baseline training DataFrame.
            numerical_features: List of continuous / numerical feature column names.
            categorical_features: List of categorical / binary feature column names.
            alpha: Significance level for KS tests (default: 0.05).
        """
        self.ref_df = reference_df.copy()
        self.alpha = alpha

        if numerical_features is None:
            # Auto-infer numerical features
            num_cols = self.ref_df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target and raw IDs if present
            excluded = {"sales", "id", "item_id", "store_id", "state_id", "cat_id", "dept_id"}
            self.numerical_features = [c for c in num_cols if c not in excluded]
        else:
            self.numerical_features = [f for f in numerical_features if f in self.ref_df.columns]

        if categorical_features is None:
            cat_cols = self.ref_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            self.categorical_features = [c for c in cat_cols if c in self.ref_df.columns]
        else:
            self.categorical_features = [f for f in categorical_features if f in self.ref_df.columns]

        logger.info(
            f"Initialized DataDriftDetector with {len(self.ref_df)} reference rows: "
            f"{len(self.numerical_features)} numerical, {len(self.categorical_features)} categorical features."
        )

    def evaluate_feature_drift(
        self,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Evaluates covariate shift across all configured features between reference and current.

        Args:
            current_df: Current operational live DataFrame.
            features: Optional subset of features to evaluate (defaults to all configured).

        Returns:
            DataFrame summarizing PSI, KS tests, Wasserstein distance, and drift status.
        """
        results = []
        target_features = features or (self.numerical_features + self.categorical_features)

        for feat in target_features:
            if feat not in current_df.columns or feat not in self.ref_df.columns:
                continue

            ref_vals = self.ref_df[feat].dropna().values
            curr_vals = current_df[feat].dropna().values

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            if feat in self.numerical_features and pd.api.types.is_numeric_dtype(self.ref_df[feat]):
                # Numerical feature tests
                psi, _ = calculate_psi(ref_vals, curr_vals)
                ks_res = kolmogorov_smirnov_test(ref_vals, curr_vals, alpha=self.alpha)
                w_res = wasserstein_distance_test(ref_vals, curr_vals)

                res = FeatureDriftResult(
                    feature_name=feat,
                    feature_type="numerical",
                    psi=psi,
                    psi_status=classify_psi(psi),
                    ks_statistic=ks_res["statistic"],
                    ks_p_value=ks_res["p_value"],
                    ks_drift_detected=ks_res["drift_detected"],
                    wasserstein_dist=w_res["wasserstein_dist"],
                    relative_wasserstein=w_res["relative_wasserstein"],
                    sample_size_ref=len(ref_vals),
                    sample_size_curr=len(curr_vals),
                )
            else:
                # Categorical feature tests
                cat_res = categorical_divergence_test(ref_vals, curr_vals)
                res = FeatureDriftResult(
                    feature_name=feat,
                    feature_type="categorical",
                    psi=cat_res["psi"],
                    psi_status=cat_res["psi_status"],
                    js_divergence=cat_res["js_divergence"],
                    sample_size_ref=len(ref_vals),
                    sample_size_curr=len(curr_vals),
                )

            results.append(res.to_dict())

        return pd.DataFrame(results)

    def evaluate_target_drift(
        self,
        current_df: pd.DataFrame,
        target_col: str = "sales",
        group_col: Optional[str] = "id",
    ) -> Dict[str, Any]:
        """Evaluates target demand volume, sparsity, and Syntetos-Boylan demand regime shift.

        Args:
            current_df: Current operational live DataFrame.
            target_col: Sales / demand column name.
            group_col: Optional series identifier for portfolio-wide regime shifts.

        Returns:
            Dictionary detailing overall target shift and individual SKU regime changes.
        """
        ref_sales = self.ref_df[target_col].dropna().values
        curr_sales = current_df[target_col].dropna().values

        # 1. Distribution shift
        psi_target, _ = calculate_psi(ref_sales, curr_sales)
        ks_target = kolmogorov_smirnov_test(ref_sales, curr_sales, alpha=self.alpha)

        # 2. Moments & Sparsity
        mean_ref = float(np.mean(ref_sales))
        mean_curr = float(np.mean(curr_sales))
        mean_pct_change = ((mean_curr - mean_ref) / (mean_ref + 1e-4)) * 100.0

        zero_ref = float(np.mean(ref_sales == 0.0))
        zero_curr = float(np.mean(curr_sales == 0.0))
        zero_pct_point_shift = (zero_curr - zero_ref) * 100.0

        # 3. Non-parametric rank-sum test
        u_stat, u_p_val = stats.mannwhitneyu(ref_sales, curr_sales, alternative="two-sided")

        # 4. Regime shift tracking (if group_col is present)
        regime_changes = []
        if group_col and group_col in self.ref_df.columns and group_col in current_df.columns:
            common_ids = list(set(self.ref_df[group_col]).intersection(set(current_df[group_col])))
            for s_id in common_ids:
                s_ref = self.ref_df.loc[self.ref_df[group_col] == s_id, target_col].values
                s_curr = current_df.loc[current_df[group_col] == s_id, target_col].values
                if len(s_ref) < 14 or len(s_curr) < 14:
                    continue

                prof_ref = calculate_demand_profile(pd.Series(s_ref))
                prof_curr = calculate_demand_profile(pd.Series(s_curr))

                if prof_ref.regime != prof_curr.regime:
                    regime_changes.append({
                        "series_id": s_id,
                        "reference_regime": prof_ref.regime,
                        "current_regime": prof_curr.regime,
                        "ref_adi": round(prof_ref.adi, 2),
                        "curr_adi": round(prof_curr.adi, 2),
                        "ref_cv2": round(prof_ref.cv2, 2),
                        "curr_cv2": round(prof_curr.cv2, 2),
                    })

        return {
            "target_column": target_col,
            "target_psi": psi_target,
            "target_psi_status": classify_psi(psi_target),
            "ks_statistic": ks_target["statistic"],
            "ks_p_value": ks_target["p_value"],
            "ks_drift_detected": ks_target["drift_detected"],
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p_value": round(float(u_p_val), 6),
            "reference_mean": round(mean_ref, 3),
            "current_mean": round(mean_curr, 3),
            "mean_percentage_change": round(mean_pct_change, 2),
            "reference_zero_ratio": round(zero_ref, 4),
            "current_zero_ratio": round(zero_curr, 4),
            "zero_ratio_shift_points": round(zero_pct_point_shift, 2),
            "regime_change_count": len(regime_changes),
            "regime_changes": regime_changes,
        }
