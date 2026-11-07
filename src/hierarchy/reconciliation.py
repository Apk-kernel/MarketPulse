"""Hierarchical forecast reconciliation engine implementing BU, TD, OLS, WLS, and MinT."""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import linalg, optimize, sparse
from sklearn.covariance import LedoitWolf

from src.data.hierarchy import M5Hierarchy
from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)
from src.hierarchy.aggregation import HierarchicalAggregator
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Hierarchy.Reconciliation")

ReconciliationMethod = Literal["bu", "td", "ols", "wls_str", "wls_var", "mint_shrink", "mint_sample"]


class HierarchicalReconciler:
    """Linear and non-negative forecast reconciliation engine for retail hierarchies.

    Implements:
    - Bottom-Up (BU)
    - Top-Down (TD)
    - Ordinary Least Squares (OLS)
    - Weighted Least Squares - Structural (WLS-Str)
    - Weighted Least Squares - Variance (WLS-Var)
    - Minimum Trace with Shrinkage (MinT-Shrink)
    - Non-negative projection via NNLS (scipy.optimize.nnls)
    """

    def __init__(
        self,
        hierarchy: M5Hierarchy,
        method: ReconciliationMethod = "mint_shrink",
        non_negative: bool = True,
        shrinkage_intensity: Optional[float] = None,
    ):
        """Initializes reconciler.

        Args:
            hierarchy: Configured M5Hierarchy instance.
            method: Reconciliation method: 'bu', 'td', 'ols', 'wls_str', 'wls_var', 'mint_shrink', 'mint_sample'.
            non_negative: If True, enforces y_tilde >= 0 via Non-Negative Least Squares (NNLS).
            shrinkage_intensity: Optional fixed shrinkage lambda in [0, 1]. If None, Ledoit-Wolf is used.
        """
        self.hierarchy = hierarchy
        self.method = method
        self.non_negative = non_negative
        self.shrinkage_intensity = shrinkage_intensity

        self.aggregator = HierarchicalAggregator(hierarchy)
        self.s_matrix, self.total_node_names, self.bottom_node_names = (
            hierarchy.build_summing_matrix()
        )
        self.S = self.s_matrix.toarray().astype(np.float64)  # Shape: (m, n)
        self.m, self.n = self.S.shape

        self._p_matrix: Optional[np.ndarray] = None  # Projection matrix P of shape (n, m)
        self._w_matrix: Optional[np.ndarray] = None  # Covariance weighting matrix W of shape (m, m)
        self._historical_props: Optional[np.ndarray] = None
        self.shrinkage_intensity_: Optional[float] = None
        self._fitted = False

    @staticmethod
    def compute_schafer_strimmer_shrinkage(
        residuals: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Computes optimal Ledoit-Wolf / Schäfer-Strimmer shrinkage toward diagonal target.

        Target: T = diag(S_sample) (diagonal sample variance matrix).
        Formula:
            S_shrink = (1 - lambda*) S_sample + lambda* diag(S_sample)
        where lambda* minimizes expected Frobenius loss under unconstrained cross-series covariance.

        Args:
            residuals: Residuals array of shape (T, m).

        Returns:
            Tuple of:
            - shrunk_cov: Positive-definite covariance matrix of shape (m, m).
            - optimal_lambda: Float shrinkage parameter in [0, 1].
        """
        T, m = residuals.shape
        if T <= 2:
            var_diag = np.var(residuals, axis=0, ddof=1)
            var_diag = np.where(var_diag < 1e-8, 1e-4, var_diag)
            return np.diag(var_diag), 1.0

        res_centered = residuals - residuals.mean(axis=0)
        sample_cov = (res_centered.T @ res_centered) / (T - 1)
        variances = np.diag(sample_cov)
        variances = np.where(variances < 1e-8, 1e-4, variances)
        np.fill_diagonal(sample_cov, variances)

        # Vectorized variance of sample covariance elements: sum_t (e_ti * e_tj)^2
        res_sq = res_centered ** 2
        sum_sq = res_sq.T @ res_sq  # (m, m)
        sum_diff_sq = sum_sq - (T - 2) * (sample_cov ** 2)
        var_s = (T / ((T - 1) ** 3)) * sum_diff_sq

        off_diag_mask = ~np.eye(m, dtype=bool)
        numerator = float(np.sum(var_s[off_diag_mask]))
        denominator = float(np.sum((sample_cov[off_diag_mask]) ** 2))

        lam = float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator > 0 else 1.0
        shrunk_cov = (1.0 - lam) * sample_cov
        np.fill_diagonal(shrunk_cov, variances)

        return shrunk_cov, lam

    @staticmethod
    def verify_reconciler_idempotency(
        S: np.ndarray,
        P: np.ndarray,
        tolerance: float = 1e-5,
    ) -> bool:
        """Mathematically verifies idempotency condition: S * P * S == S.

        An optimal linear reconciler must leave already-coherent forecasts unchanged.
        """
        s_p_s = S @ P @ S
        max_diff = float(np.max(np.abs(s_p_s - S)))
        return max_diff <= tolerance

    @staticmethod
    def verify_covariance_matrix(
        W: np.ndarray,
        min_eigenvalue_tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """Mathematically verifies covariance matrix symmetry and positive definiteness."""
        is_symmetric = bool(np.allclose(W, W.T, atol=1e-8))
        evals = np.linalg.eigvalsh(W)
        min_eval = float(np.min(evals))
        max_eval = float(np.max(evals))
        is_positive_definite = bool(min_eval >= min_eigenvalue_tol)
        condition_number = float(max_eval / min_eval) if min_eval > 0 else float("inf")

        return {
            "is_symmetric": is_symmetric,
            "is_positive_definite": is_positive_definite,
            "min_eigenvalue": min_eval,
            "max_eigenvalue": max_eval,
            "condition_number": condition_number,
        }

    def fit(
        self,
        in_sample_actuals: Optional[pd.DataFrame] = None,
        in_sample_base_forecasts: Optional[pd.DataFrame] = None,
    ) -> "HierarchicalReconciler":
        """Fits the reconciliation weighting matrix using in-sample residuals.

        Args:
            in_sample_actuals: DataFrame of in-sample actual values (shape T x m or T x n).
            in_sample_base_forecasts: DataFrame of in-sample base forecasts across all m nodes (shape T x m).

        Returns:
            self
        """
        logger.info(f"Fitting HierarchicalReconciler with method='{self.method}', non_negative={self.non_negative}")

        if self.method == "bu":
            self._fit_bottom_up()

        elif self.method == "td":
            if in_sample_actuals is None:
                raise ValueError("Method 'td' requires in_sample_actuals to compute historical proportions.")
            self._fit_top_down(in_sample_actuals)

        elif self.method == "ols":
            self._fit_ols()

        elif self.method == "wls_str":
            self._fit_wls_structural()

        elif self.method in ("wls_var", "mint_shrink", "mint_sample"):
            if in_sample_actuals is None or in_sample_base_forecasts is None:
                raise ValueError(
                    f"Method '{self.method}' requires both in_sample_actuals and in_sample_base_forecasts."
                )
            self._fit_mint(in_sample_actuals, in_sample_base_forecasts)

        else:
            raise ValueError(f"Unknown reconciliation method: {self.method}")

        self._fitted = True
        return self

    def _fit_bottom_up(self) -> None:
        """Constructs Bottom-Up projection matrix P_bu: extracts bottom rows."""
        # Bottom nodes are indexed by their positions in total_node_names
        bottom_indices = [self.total_node_names.index(name) for name in self.bottom_node_names]
        P = np.zeros((self.n, self.m), dtype=np.float64)
        for i, idx in enumerate(bottom_indices):
            P[i, idx] = 1.0
        self._p_matrix = P

    def _fit_top_down(self, actuals_df: pd.DataFrame) -> None:
        """Constructs Top-Down projection matrix using historical sales proportions."""
        # Ensure we have bottom series
        if set(self.bottom_node_names).issubset(set(actuals_df.columns)):
            bottom_df = actuals_df[self.bottom_node_names]
        else:
            bottom_df = self.aggregator.prepare_bottom_wide_dataframe(actuals_df)

        props = self.aggregator.compute_historical_proportions(bottom_df, method="pha")
        self._historical_props = props

        # Top node 'Total' is index 0
        total_idx = self.total_node_names.index("Total") if "Total" in self.total_node_names else 0
        P = np.zeros((self.n, self.m), dtype=np.float64)
        P[:, total_idx] = props
        self._p_matrix = P

    def _fit_ols(self) -> None:
        """P_ols = (S^T S)^(-1) S^T."""
        sts = self.S.T @ self.S  # (n, n)
        sts_inv = linalg.pinv(sts)
        self._p_matrix = sts_inv @ self.S.T  # (n, m)
        self._w_matrix = np.eye(self.m, dtype=np.float64)

    def _fit_wls_structural(self) -> None:
        """W_str = diag(S * 1_n), P = (S^T W^-1 S)^(-1) S^T W^-1."""
        node_counts = self.S.sum(axis=1)  # (m,)
        w_diag = np.maximum(node_counts, 1.0)
        w_inv_diag = 1.0 / w_diag
        self._w_matrix = np.diag(w_diag)

        # S^T W^-1 S = S^T @ diag(w_inv) @ S
        st_winv = self.S.T * w_inv_diag  # broadcasting rows: (n, m)
        st_winv_s = st_winv @ self.S  # (n, n)
        st_winv_s_inv = linalg.pinv(st_winv_s)
        self._p_matrix = st_winv_s_inv @ st_winv  # (n, m)

    def _fit_mint(
        self,
        actuals_df: pd.DataFrame,
        base_forecasts_df: pd.DataFrame,
    ) -> None:
        """Fits MinT weighting matrix using in-sample forecast errors."""
        # Align columns
        cols = self.total_node_names
        common_cols = [c for c in cols if c in actuals_df.columns and c in base_forecasts_df.columns]
        if len(common_cols) != len(cols):
            missing = set(cols) - set(common_cols)
            raise ValueError(f"Actuals or base forecasts missing {len(missing)} hierarchy nodes.")

        # Compute residuals: E = Actuals - Forecasts (T x m)
        residuals = (actuals_df[cols].values - base_forecasts_df[cols].values).astype(np.float64)
        T, m = residuals.shape

        # Remove zero variance / constant columns for stable covariance
        variances = np.var(residuals, axis=0)
        variances = np.where(variances < 1e-8, 1e-4, variances)

        if self.method == "wls_var":
            # Diagonal variance weighting
            W = np.diag(variances)
            w_inv_diag = 1.0 / variances
            st_winv = self.S.T * w_inv_diag
            st_winv_s_inv = linalg.pinv(st_winv @ self.S)
            self._p_matrix = st_winv_s_inv @ st_winv
            self._w_matrix = W

        elif self.method == "mint_sample":
            # Sample covariance matrix
            sample_cov = np.cov(residuals, rowvar=False)
            # Add small Tikhonov ridge for numerical invertibility
            reg_cov = sample_cov + 1e-4 * np.eye(m)
            self._w_matrix = reg_cov
            w_inv = linalg.pinv(reg_cov)
            st_winv_s_inv = linalg.pinv(self.S.T @ w_inv @ self.S)
            self._p_matrix = st_winv_s_inv @ self.S.T @ w_inv

        elif self.method == "mint_shrink":
            # Schäfer-Strimmer / Ledoit-Wolf diagonal target shrinkage
            if self.shrinkage_intensity is not None:
                lam = float(np.clip(self.shrinkage_intensity, 0.0, 1.0))
                res_centered = residuals - residuals.mean(axis=0)
                sample_cov = (res_centered.T @ res_centered) / max(1, T - 1)
                target = np.diag(variances)
                shrunk_cov = (1.0 - lam) * sample_cov
                np.fill_diagonal(shrunk_cov, variances)
            else:
                shrunk_cov, lam = self.compute_schafer_strimmer_shrinkage(residuals)

            self.shrinkage_intensity_ = lam
            # Ridge regularization for strictly positive eigenvalues
            shrunk_cov += 1e-4 * np.eye(m)
            self._w_matrix = shrunk_cov
            w_inv = linalg.pinv(shrunk_cov)
            st_winv_s_inv = linalg.pinv(self.S.T @ w_inv @ self.S)
            self._p_matrix = st_winv_s_inv @ self.S.T @ w_inv

    def reconcile(
        self,
        base_forecasts_wide_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Reconciles base forecasts across all hierarchy nodes into coherent forecasts.

        Args:
            base_forecasts_wide_df: DataFrame indexed by horizon/date with columns matching total_node_names.

        Returns:
            DataFrame of reconciled coherent forecasts with the same index and columns.
        """
        if not self._fitted:
            raise RuntimeError("HierarchicalReconciler must be fitted before calling reconcile().")

        cols = self.total_node_names
        missing = set(cols) - set(base_forecasts_wide_df.columns)
        if missing:
            raise ValueError(f"base_forecasts_wide_df missing {len(missing)} nodes.")

        # y_hat matrix: (H, m)
        y_hat = base_forecasts_wide_df[cols].values.astype(np.float64)
        H, m = y_hat.shape

        if self.non_negative:
            # Solve non-negative reconciliation
            y_tilde = self._reconcile_non_negative(y_hat)
        else:
            # Standard unconstrained linear reconciliation: y_tilde = y_hat @ P.T @ S.T
            y_bottom_reconciled = y_hat @ self._p_matrix.T  # (H, n)
            y_tilde = y_bottom_reconciled @ self.S.T  # (H, m)

        reconciled_df = pd.DataFrame(
            y_tilde,
            index=base_forecasts_wide_df.index,
            columns=cols,
        )

        return reconciled_df

    def _reconcile_non_negative(self, y_hat: np.ndarray) -> np.ndarray:
        """Enforces non-negativity constraint y_tilde_bottom >= 0 via NNLS or non-negative projection.

        - Bottom-Up (BU): y_tilde_b = max(0, y_hat_b), y_tilde = S @ y_tilde_b
        - Top-Down (TD): y_tilde_b = props * max(0, y_hat_Total), y_tilde = S @ y_tilde_b
        - Least Squares / MinT: min_{y_b >= 0} || W^(-1/2) S y_b - W^(-1/2) y_hat_h ||_2^2
        """
        H, m = y_hat.shape

        if self.method == "bu":
            bottom_indices = [self.total_node_names.index(name) for name in self.bottom_node_names]
            y_bottom = np.maximum(0.0, y_hat[:, bottom_indices])  # (H, n)
            return y_bottom @ self.S.T  # (H, m)

        elif self.method == "td":
            total_idx = self.total_node_names.index("Total") if "Total" in self.total_node_names else 0
            props = self._historical_props if self._historical_props is not None else np.ones(self.n) / self.n
            y_top = np.maximum(0.0, y_hat[:, total_idx:total_idx + 1])  # (H, 1)
            y_bottom = y_top @ props.reshape(1, -1)  # (H, n)
            return y_bottom @ self.S.T  # (H, m)

        # For OLS, WLS, and MinT: solve non-negative least squares per horizon step
        y_tilde = np.zeros((H, m), dtype=np.float64)

        # Prepare weighting matrix square root W^(-1/2)
        if self._w_matrix is not None:
            try:
                # Use eigenvalue decomposition for symmetric positive-definite inverse square root
                evals, evecs = np.linalg.eigh(self._w_matrix)
                evals = np.maximum(evals, 1e-6)
                w_inv_half = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            except Exception:
                w_diag = np.diag(self._w_matrix)
                w_inv_half = np.diag(1.0 / np.sqrt(np.maximum(w_diag, 1e-6)))
        else:
            w_inv_half = np.eye(m, dtype=np.float64)

        # A = W^(-1/2) @ S: shape (m, n)
        A = w_inv_half @ self.S

        for h in range(H):
            # b = W^(-1/2) @ y_hat[h]: shape (m,)
            b = w_inv_half @ y_hat[h]

            # Solve min_{x >= 0} || A x - b ||_2
            x_opt, _ = optimize.nnls(A, b)

            # Reconstructed coherent non-negative forecast across all nodes: S @ x_opt
            y_tilde[h] = self.S @ x_opt

        return y_tilde

    @staticmethod
    def evaluate_reconciliation_impact(
        base_forecasts: pd.DataFrame,
        reconciled_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: M5Hierarchy,
    ) -> pd.DataFrame:
        """Compares forecast accuracy before and after reconciliation across all 12 hierarchy levels.

        Args:
            base_forecasts: Base forecasts wide DataFrame.
            reconciled_forecasts: Reconciled forecasts wide DataFrame.
            actuals: Ground truth wide DataFrame.
            hierarchy: M5Hierarchy instance.

        Returns:
            DataFrame summarizing MAE, RMSE, sMAPE, and WAPE before and after reconciliation per level.
        """
        records = []

        for lvl in range(1, 13):
            lvl_name = hierarchy.LEVEL_DEFINITIONS[lvl]["name"]
            lvl_nodes = [node for node in hierarchy.get_level_nodes(lvl) if node in actuals.columns]

            if not lvl_nodes:
                continue

            y_act = actuals[lvl_nodes].values.ravel()
            y_base = base_forecasts[lvl_nodes].values.ravel()
            y_rec = reconciled_forecasts[lvl_nodes].values.ravel()

            base_mae = mean_absolute_error(y_act, y_base)
            rec_mae = mean_absolute_error(y_act, y_rec)

            base_rmse = root_mean_squared_error(y_act, y_base)
            rec_rmse = root_mean_squared_error(y_act, y_rec)

            base_smape = symmetric_mean_absolute_percentage_error(y_act, y_base)
            rec_smape = symmetric_mean_absolute_percentage_error(y_act, y_rec)

            base_wape = weighted_absolute_percentage_error(y_act, y_base)
            rec_wape = weighted_absolute_percentage_error(y_act, y_rec)

            mae_improvement = ((base_mae - rec_mae) / base_mae) * 100.0 if base_mae > 0 else 0.0
            rmse_improvement = ((base_rmse - rec_rmse) / base_rmse) * 100.0 if base_rmse > 0 else 0.0
            wape_improvement = ((base_wape - rec_wape) / base_wape) * 100.0 if base_wape > 0 else 0.0

            records.append({
                "level": lvl,
                "level_name": lvl_name,
                "num_series": len(lvl_nodes),
                "base_mae": round(base_mae, 4),
                "rec_mae": round(rec_mae, 4),
                "mae_improvement_pct": round(mae_improvement, 2),
                "base_rmse": round(base_rmse, 4),
                "rec_rmse": round(rec_rmse, 4),
                "rmse_improvement_pct": round(rmse_improvement, 2),
                "base_wape": round(base_wape, 4),
                "rec_wape": round(rec_wape, 4),
                "wape_improvement_pct": round(wape_improvement, 2),
                "base_smape": round(base_smape, 4),
                "rec_smape": round(rec_smape, 4),
            })

        return pd.DataFrame(records)

