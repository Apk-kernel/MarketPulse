"""Formal research experiments engine for MarketPulse.

Implements the 6 core research benchmarks:
1. Local vs Global Forecasting
2. Value of Forecast Reconciliation (Homogeneous Base Model)
3. Exogenous Promotional & Pricing Feature Ablation
4. Cold-Start Demand Estimation
5. Intermittent Demand Regime Routing
6. Forecast Horizon Deterioration Dynamics
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

from src.data.hierarchy import M5Hierarchy
from src.features.demand_features import calculate_demand_profile
from src.features.lag_features import create_lag_features
from src.features.rolling_features import create_rolling_features
from src.forecasting.baselines import create_seasonal_naive, create_theta_forecaster
from src.forecasting.probabilistic import IntermittentProbabilisticForecaster
from src.hierarchy.aggregation import HierarchicalAggregator
from src.hierarchy.hierarchy_builder import HierarchyTree
from src.hierarchy.reconciliation import HierarchicalReconciler
from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_scaled_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
    weighted_root_mean_squared_scaled_error,
)
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Experiments")


class ResearchExperimentRunner:
    """Orchestrates and executes formal research experiments for MarketPulse."""

    def __init__(
        self,
        processed_df: pd.DataFrame,
        horizon: int = 28,
        n_origins: int = 1,
        origin_step: Optional[int] = None,
    ):
        """Initializes experiment runner.

        Args:
            processed_df: Fully preprocessed M5 DataFrame.
            horizon: Primary evaluation horizon (default: 28 days).
            n_origins: Number of rolling forecast origins (default: 1 for standard single-cut benchmark).
            origin_step: Step size in days between successive origins (defaults to horizon).
        """
        self.df = processed_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.horizon = horizon
        self.n_origins = max(1, n_origins)
        self.origin_step = origin_step or horizon

        self.tree = HierarchyTree(metadata_df=self.df)
        self.hierarchy = self.tree.m5_hierarchy
        self.aggregator = HierarchicalAggregator(hierarchy=self.hierarchy)

        self.bottom_wide = self.aggregator.prepare_bottom_wide_dataframe(self.df)
        self.all_wide = self.aggregator.aggregate_bottom_to_all_levels(self.bottom_wide)

        # Ensure daily frequency
        dates = pd.date_range(start=self.all_wide.index[0], periods=len(self.all_wide), freq="D")
        self.all_wide.index = dates
        self.bottom_wide.index = dates

        self.T_total = len(self.all_wide)
        self.train_size = self.T_total - self.horizon

    @property
    def train_dates(self) -> pd.DatetimeIndex:
        """Returns the DatetimeIndex for the active training window."""
        return self.all_wide.index[: self.train_size]

    @property
    def test_dates(self) -> pd.DatetimeIndex:
        """Returns the DatetimeIndex for the active test evaluation window (length = horizon)."""
        return self.all_wide.index[self.train_size : self.train_size + self.horizon]

    def get_rolling_origins(self) -> List[Dict[str, Any]]:
        """Generates rolling temporal evaluation cutoffs across the time series.

        Returns:
            List of dictionaries specifying origin index, train/test boundaries, and window ranges.
        """
        origins = []
        for i in range(self.n_origins):
            offset = (self.n_origins - 1 - i) * self.origin_step
            test_end_idx = self.T_total - offset
            train_end_idx = test_end_idx - self.horizon
            if train_end_idx < self.horizon * 2:
                continue

            origins.append({
                "origin_idx": i + 1,
                "train_size": train_end_idx,
                "test_end_idx": test_end_idx,
                "train_start": str(self.all_wide.index[0].date()),
                "train_end": str(self.all_wide.index[train_end_idx - 1].date()),
                "test_start": str(self.all_wide.index[train_end_idx].date()),
                "test_end": str(self.all_wide.index[test_end_idx - 1].date()),
                "horizon": self.horizon,
            })
        return origins or [{
            "origin_idx": 1,
            "train_size": self.train_size,
            "test_end_idx": self.T_total,
            "train_start": str(self.all_wide.index[0].date()),
            "train_end": str(self.all_wide.index[self.train_size - 1].date()),
            "test_start": str(self.all_wide.index[self.train_size].date()),
            "test_end": str(self.all_wide.index[self.train_size + self.horizon - 1].date()),
            "horizon": self.horizon,
        }]

    def run_multi_origin_experiment(
        self,
        experiment_func_name: str = "run_experiment_1_local_vs_global",
    ) -> pd.DataFrame:
        """Executes a specified research experiment across all configured rolling origins.

        Args:
            experiment_func_name: Method name on self to execute across origins.

        Returns:
            Concatenated DataFrame with an added 'origin_idx' column.
        """
        origins = self.get_rolling_origins()
        if len(origins) == 1:
            func = getattr(self, experiment_func_name)
            res = func()
            res["origin_idx"] = 1
            return res

        all_results = []
        original_train_size = self.train_size
        try:
            for orig in origins:
                self.train_size = orig["train_size"]
                func = getattr(self, experiment_func_name)
                df_res = func()
                df_res["origin_idx"] = orig["origin_idx"]
                df_res["train_end"] = orig["train_end"]
                df_res["test_start"] = orig["test_start"]
                all_results.append(df_res)
        finally:
            self.train_size = original_train_size

        return pd.concat(all_results, ignore_index=True)

    # =========================================================================
    # Experiment 1: Local vs Global Forecasting
    # =========================================================================
    def run_experiment_1_local_vs_global(self) -> pd.DataFrame:
        """Compares individual series models (Local) against pooled cross-learning models (Global)."""
        logger.info("Executing Experiment 1: Local vs Global Forecasting...")
        cols = self.bottom_wide.columns
        y_train = self.bottom_wide.iloc[: self.train_size]
        y_test = self.bottom_wide.iloc[self.train_size : self.train_size + self.horizon]

        # 1. Local Models: Separate LightGBM per series
        local_preds = np.zeros((self.horizon, len(cols)), dtype=np.float64)
        local_start = time.perf_counter()

        for j, col in enumerate(cols):
            s = y_train[col]
            df_s = pd.DataFrame({"sales": s.values}, index=s.index)
            df_feat = create_lag_features(df_s, target_col="sales", lags=[1, 7, 14])
            df_feat = create_rolling_features(
                df_feat, target_col="sales", mean_windows=[7, 14], std_windows=[], shift=1
            )
            df_feat["dayofweek"] = df_feat.index.dayofweek
            clean = df_feat.dropna()

            X = clean.drop(columns=["sales"])
            y = clean["sales"]

            reg = lgb.LGBMRegressor(n_estimators=40, learning_rate=0.08, verbose=-1, random_state=42)
            reg.fit(X, y)

            # Recursive forecast
            hist = list(s.values)
            for h in range(self.horizon):
                d_idx = self.test_dates[h]
                feat_dict = {
                    "lag_1": hist[-1],
                    "lag_7": hist[-7] if len(hist) >= 7 else hist[-1],
                    "lag_14": hist[-14] if len(hist) >= 14 else hist[-1],
                    "rolling_mean_7": np.mean(hist[-7:]),
                    "rolling_mean_14": np.mean(hist[-14:]),
                    "dayofweek": d_idx.dayofweek,
                }
                pred_val = float(reg.predict(pd.DataFrame([feat_dict]))[0])
                pred_val = max(0.0, pred_val)
                local_preds[h, j] = pred_val
                hist.append(pred_val)

        local_time = time.perf_counter() - local_start

        # 2. Global Model: Pooled panel across all 45 series
        global_start = time.perf_counter()
        records_train = []
        for col in cols:
            s = y_train[col]
            df_s = pd.DataFrame({"sales": s.values}, index=s.index)
            df_feat = create_lag_features(df_s, target_col="sales", lags=[1, 7, 14])
            df_feat = create_rolling_features(
                df_feat, target_col="sales", mean_windows=[7, 14], std_windows=[], shift=1
            )
            df_feat["dayofweek"] = df_feat.index.dayofweek
            df_feat["series_id"] = col
            records_train.append(df_feat)

        global_train_df = pd.concat(records_train).dropna()
        cat_type = pd.CategoricalDtype(categories=list(cols))
        global_train_df["series_id"] = global_train_df["series_id"].astype(cat_type)

        X_glob = global_train_df.drop(columns=["sales"])
        y_glob = global_train_df["sales"]

        glob_reg = lgb.LGBMRegressor(n_estimators=80, learning_rate=0.08, verbose=-1, random_state=42)
        glob_reg.fit(X_glob, y_glob)

        global_preds = np.zeros((self.horizon, len(cols)), dtype=np.float64)
        for j, col in enumerate(cols):
            hist = list(y_train[col].values)
            for h in range(self.horizon):
                d_idx = self.test_dates[h]
                feat_dict = {
                    "lag_1": hist[-1],
                    "lag_7": hist[-7] if len(hist) >= 7 else hist[-1],
                    "lag_14": hist[-14] if len(hist) >= 14 else hist[-1],
                    "rolling_mean_7": np.mean(hist[-7:]),
                    "rolling_mean_14": np.mean(hist[-14:]),
                    "dayofweek": d_idx.dayofweek,
                    "series_id": col,
                }
                pred_df = pd.DataFrame([feat_dict])
                pred_df["series_id"] = pred_df["series_id"].astype(cat_type)
                pred_val = float(glob_reg.predict(pred_df)[0])
                pred_val = max(0.0, pred_val)
                global_preds[h, j] = pred_val
                hist.append(pred_val)

        global_time = time.perf_counter() - global_start

        # Stratified evaluation by volume tier
        mean_vols = y_train.mean(axis=0)
        high_vol_mask = mean_vols >= mean_vols.quantile(0.67)
        low_vol_mask = ~high_vol_mask

        results = []
        for tier_name, mask in [("Overall (All SKUs)", slice(None)), ("High-Volume Tier (Top 33%)", high_vol_mask), ("Low-Volume Tier (Bottom 67%)", low_vol_mask)]:
            act_sub = y_test.values[:, mask].ravel()
            loc_sub = local_preds[:, mask].ravel()
            glob_sub = global_preds[:, mask].ravel()

            loc_wape = weighted_absolute_percentage_error(act_sub, loc_sub)
            glob_wape = weighted_absolute_percentage_error(act_sub, glob_sub)
            loc_rmse = root_mean_squared_error(act_sub, loc_sub)
            glob_rmse = root_mean_squared_error(act_sub, glob_sub)

            wape_gain = ((loc_wape - glob_wape) / loc_wape) * 100.0 if loc_wape > 0 else 0.0

            results.append({
                "Evaluation Tier": tier_name,
                "Local WAPE (%)": round(loc_wape, 2),
                "Global WAPE (%)": round(glob_wape, 2),
                "Global Improvement (%)": round(wape_gain, 2),
                "Local RMSE": round(loc_rmse, 3),
                "Global RMSE": round(glob_rmse, 3),
                "Local Time (s)": round(local_time, 2),
                "Global Time (s)": round(global_time, 2),
            })

        return pd.DataFrame(results)

    # =========================================================================
    # Experiment 2: Value of Forecast Reconciliation
    # =========================================================================
    def run_experiment_2_reconciliation_value(self) -> pd.DataFrame:
        """Quantifies accuracy impact of reconciliation using an equalized base model family."""
        logger.info("Executing Experiment 2: Value of Forecast Reconciliation...")
        y_train_all = self.all_wide.iloc[: self.train_size]
        y_test_all = self.all_wide.iloc[self.train_size : self.train_size + self.horizon]
        cols = self.all_wide.columns
        m = len(cols)

        # Fit homogeneous LightGBM point forecasters for ALL nodes
        base_preds = np.zeros((self.horizon, m), dtype=np.float64)
        fitted_in_sample = np.zeros((self.train_size, m), dtype=np.float64)

        for j, col in enumerate(cols):
            s = y_train_all[col]
            df_s = pd.DataFrame({"sales": s.values}, index=s.index)
            df_feat = create_lag_features(df_s, target_col="sales", lags=[1, 7])
            df_feat = create_rolling_features(
                df_feat, target_col="sales", mean_windows=[7], std_windows=[], shift=1
            )
            df_feat["dayofweek"] = df_feat.index.dayofweek
            clean = df_feat.dropna()

            X = clean.drop(columns=["sales"])
            y = clean["sales"]

            reg = lgb.LGBMRegressor(n_estimators=30, learning_rate=0.1, verbose=-1, random_state=42)
            reg.fit(X, y)

            # In-sample fitted
            fitted_in_sample[len(df_s) - len(clean):, j] = reg.predict(X)

            # Recursive predict
            hist = list(s.values)
            for h in range(self.horizon):
                d_idx = self.test_dates[h]
                feat = {
                    "lag_1": hist[-1],
                    "lag_7": hist[-7] if len(hist) >= 7 else hist[-1],
                    "rolling_mean_7": np.mean(hist[-7:]),
                    "dayofweek": d_idx.dayofweek,
                }
                pred_v = max(0.0, float(reg.predict(pd.DataFrame([feat]))[0]))
                base_preds[h, j] = pred_v
                hist.append(pred_v)

        base_df = pd.DataFrame(base_preds, index=self.test_dates, columns=cols)
        in_sample_df = pd.DataFrame(fitted_in_sample, index=self.train_dates, columns=cols)

        # Initial incoherence
        incoh_base = self.aggregator.verify_coherence(base_df)

        methods = [
            ("Base (Unreconciled)", None, False),
            ("Bottom-Up (BU)", "bu", False),
            ("Top-Down (TD-PHA)", "td", False),
            ("Ordinary Least Squares (OLS)", "ols", False),
            ("WLS Structural (WLS-Str)", "wls_str", False),
            ("MinT Shrinkage", "mint_shrink", False),
            ("Non-Negative MinT (NN-MinT)", "mint_shrink", True),
        ]

        leaderboard = []
        for name, m_name, non_neg in methods:
            if m_name is None:
                rec_df = base_df
            else:
                reconciler = HierarchicalReconciler(
                    hierarchy=self.hierarchy, method=m_name, non_negative=non_neg
                )
                if m_name == "td":
                    reconciler.fit(in_sample_actuals=self.bottom_wide.iloc[: self.train_size])
                elif m_name in ("wls_var", "mint_shrink"):
                    reconciler.fit(
                        in_sample_actuals=y_train_all.iloc[7:],
                        in_sample_base_forecasts=in_sample_df.iloc[7:],
                    )
                else:
                    reconciler.fit()
                rec_df = reconciler.reconcile(base_df)

            coh = self.aggregator.verify_coherence(rec_df)

            act_all = y_test_all.values.ravel()
            pred_all = rec_df.values.ravel()
            overall_wape = weighted_absolute_percentage_error(act_all, pred_all)
            overall_mae = mean_absolute_error(act_all, pred_all)

            # L1 Total
            l1_wape = weighted_absolute_percentage_error(
                y_test_all["Total"].values, rec_df["Total"].values
            )

            leaderboard.append({
                "Method": name,
                "Is Coherent": "YES" if coh["is_coherent"] else "NO",
                "Max Incoherence": round(coh["max_abs_incoherence"], 4),
                "Overall WAPE (%)": round(overall_wape, 2),
                "Overall MAE": round(overall_mae, 3),
                "Total L1 WAPE (%)": round(l1_wape, 2),
            })

        return pd.DataFrame(leaderboard)

    # =========================================================================
    # Experiment 3: Exogenous Promotional & Pricing Feature Ablation
    # =========================================================================
    def run_experiment_3_exogenous_ablation(self) -> pd.DataFrame:
        """Ablation study measuring marginal gain from Calendar, Events, SNAP, and Price signals."""
        logger.info("Executing Experiment 3: Exogenous Feature Ablation Study...")
        df_feat = self.df.copy().sort_values(["id", "date"]).reset_index(drop=True)

        # Feature sets
        feature_sets = [
            ("M0: Autoregressive Lags & Rolling Only", ["lag_1", "lag_7", "rolling_mean_7"]),
            ("M1: M0 + Calendar Seasonality", ["lag_1", "lag_7", "rolling_mean_7", "dayofweek", "month"]),
            ("M2: M1 + Promotional Holiday Events", ["lag_1", "lag_7", "rolling_mean_7", "dayofweek", "month", "has_event"]),
            ("M3: M2 + SNAP Welfare Injections", ["lag_1", "lag_7", "rolling_mean_7", "dayofweek", "month", "has_event", "snap_active"]),
            ("M4: M3 + Price & Discount Dynamics (Full)", ["lag_1", "lag_7", "rolling_mean_7", "dayofweek", "month", "has_event", "snap_active", "sell_price"]),
        ]

        # Engineer features per series
        engineered_dfs = []
        for s_id, group in df_feat.groupby("id", observed=True):
            group_sorted = group.sort_values("date").copy()
            group_lags = create_lag_features(group_sorted, target_col="sales", lags=[1, 7])
            group_rolling = create_rolling_features(
                group_lags, target_col="sales", mean_windows=[7], std_windows=[], shift=1
            )
            group_rolling["dayofweek"] = group_rolling["date"].dt.dayofweek
            group_rolling["month"] = group_rolling["date"].dt.month
            group_rolling["has_event"] = group_rolling["event_name_1"].notna().astype(int)
            engineered_dfs.append(group_rolling)

        full_panel = pd.concat(engineered_dfs).dropna(
            subset=["lag_1", "lag_7", "rolling_mean_7"]
        ).reset_index(drop=True)
        split_date = self.train_dates[-1]
        test_end_date = self.test_dates[-1]

        train_mask = full_panel["date"] <= split_date
        test_mask = (full_panel["date"] > split_date) & (full_panel["date"] <= test_end_date)

        y_train = full_panel.loc[train_mask, "sales"]
        y_test = full_panel.loc[test_mask, "sales"]
        snap_test_mask = full_panel.loc[test_mask, "snap_active"] == 1

        ablation_results = []
        base_wape = None

        for name, feats in feature_sets:
            X_train = full_panel.loc[train_mask, feats]

            model = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.08, verbose=-1, random_state=42)
            model.fit(X_train, y_train)

            # Strict recursive forecasting: feed predicted values into lags for t >= 2
            preds_list = []
            test_actuals_list = []
            snap_mask_list = []

            for s_id, s_df in full_panel.groupby("id", observed=True):
                s_train = s_df[s_df["date"] <= split_date].sort_values("date")
                s_test = s_df[(s_df["date"] > split_date) & (s_df["date"] <= test_end_date)].sort_values("date")
                hist = list(s_train["sales"].values)

                for h in range(len(s_test)):
                    test_row = s_test.iloc[h]
                    d_idx = test_row["date"]
                    feat_dict = {
                        "lag_1": hist[-1],
                        "lag_7": hist[-7] if len(hist) >= 7 else hist[-1],
                        "rolling_mean_7": np.mean(hist[-7:]),
                        "dayofweek": d_idx.dayofweek,
                        "month": d_idx.month,
                        "has_event": int(test_row["has_event"]),
                        "snap_active": int(test_row["snap_active"]),
                        "sell_price": float(test_row["sell_price"]),
                    }
                    X_step = pd.DataFrame([{f: feat_dict[f] for f in feats}])
                    pred_v = max(0.0, float(model.predict(X_step)[0]))
                    preds_list.append(pred_v)
                    hist.append(pred_v)
                    test_actuals_list.append(float(test_row["sales"]))
                    snap_mask_list.append(bool(test_row["snap_active"] == 1))

            preds = np.array(preds_list)
            y_test_arr = np.array(test_actuals_list)
            snap_test_arr = np.array(snap_mask_list)

            wape_overall = weighted_absolute_percentage_error(y_test_arr, preds)
            mae_overall = mean_absolute_error(y_test_arr, preds)

            # Error specifically on SNAP benefit disbursement days
            if snap_test_arr.sum() > 0:
                wape_snap = weighted_absolute_percentage_error(
                    y_test_arr[snap_test_arr], preds[snap_test_arr]
                )
            else:
                wape_snap = wape_overall

            if base_wape is None:
                base_wape = wape_overall
                rel_gain = 0.0
            else:
                rel_gain = ((base_wape - wape_overall) / base_wape) * 100.0

            ablation_results.append({
                "Feature Set Configuration": name,
                "Features Count": len(feats),
                "Overall WAPE (%)": round(wape_overall, 2),
                "Overall MAE": round(mae_overall, 3),
                "SNAP Days WAPE (%)": round(wape_snap, 2),
                "Marginal Lift (%)": round(rel_gain, 2),
            })

        return pd.DataFrame(ablation_results)

    # =========================================================================
    # Experiment 4: Cold-Start Demand Estimation
    # =========================================================================
    def run_experiment_4_cold_start(self) -> pd.DataFrame:
        """Evaluates forecasting new products with zero sales history using hierarchical priors."""
        logger.info("Executing Experiment 4: Cold-Start Demand Estimation...")
        cols = list(self.bottom_wide.columns)
        rng = np.random.RandomState(42)

        # Select 25% of items to simulate as newly launched (zero sales in training)
        n_cold = max(2, int(len(cols) * 0.25))
        cold_cols = rng.choice(cols, size=n_cold, replace=False).tolist()
        warm_cols = [c for c in cols if c not in cold_cols]

        y_test_cold = self.bottom_wide.loc[self.test_dates, cold_cols]
        y_train_warm = self.bottom_wide.loc[self.train_dates, warm_cols]

        # Strategy 1: Naive Zero (Zero Demand Assumption)
        pred_zero = np.zeros_like(y_test_cold.values)

        # Strategy 2: Category Global Average Prior
        cat_means = {}
        for c in warm_cols:
            parts = c.split("_")
            cat = parts[2]  # Category prefix (FOODS, HOBBIES, HOUSEHOLD)
            if cat not in cat_means:
                cat_means[cat] = []
            cat_means[cat].append(y_train_warm[c].mean())

        cat_prior = {cat: float(np.mean(vals)) for cat, vals in cat_means.items()}
        global_fallback = float(y_train_warm.values.mean())

        pred_cat = np.zeros_like(y_test_cold.values)
        for j, c in enumerate(cold_cols):
            cat = c.split("_")[2]
            pred_cat[:, j] = cat_prior.get(cat, global_fallback)

        # Strategy 3: Hierarchical Prior (Department Mean + Store Seasonality Factor)
        pred_hier = np.zeros_like(y_test_cold.values)
        store_means = self.all_wide.loc[self.train_dates, self.hierarchy.get_level_nodes(3)].mean()
        state_mean = self.all_wide.loc[self.train_dates, "Total"].mean() / len(store_means)

        for j, c in enumerate(cold_cols):
            parts = c.split("_")
            store = "_".join(parts[:2])
            cat = parts[2]
            base_prior = cat_prior.get(cat, global_fallback)
            store_factor = float(store_means.get(store, state_mean) / (state_mean + 1e-4))
            pred_hier[:, j] = base_prior * store_factor

        strategies = [
            ("Strategy 1: Naive Zero Forecast", pred_zero),
            ("Strategy 2: Category Average Prior", pred_cat),
            ("Strategy 3: Hierarchical Top-Down Prior", pred_hier),
        ]

        results = []
        act_vals = y_test_cold.values.ravel()
        for name, p_mat in strategies:
            p_vals = p_mat.ravel()
            wape = weighted_absolute_percentage_error(act_vals, p_vals)
            mae = mean_absolute_error(act_vals, p_vals)
            rmse = root_mean_squared_error(act_vals, p_vals)

            results.append({
                "Cold-Start Strategy": name,
                "Evaluated SKUs": n_cold,
                "Cold-Start WAPE (%)": round(wape, 2),
                "Cold-Start MAE": round(mae, 3),
                "Cold-Start RMSE": round(rmse, 3),
            })

        return pd.DataFrame(results)

    # =========================================================================
    # Experiment 5: Intermittent Demand Regime Routing
    # =========================================================================
    def run_experiment_5_regime_routing(self) -> pd.DataFrame:
        """Tests Syntetos-Boylan demand regime routing vs universal one-size-fits-all model."""
        logger.info("Executing Experiment 5: Intermittent Demand Regime Routing...")
        cols = self.bottom_wide.columns
        y_train = self.bottom_wide.iloc[: self.train_size]
        y_test = self.bottom_wide.iloc[self.train_size : self.train_size + self.horizon]

        # Classify each series using Syntetos-Boylan (2005)
        profiles = {}
        for col in cols:
            profiles[col] = calculate_demand_profile(y_train[col])

        regimes_map = {col: p.regime for col, p in profiles.items()}
        regimes_count = pd.Series(regimes_map).value_counts()

        # Strategy A: Universal Model (Global LightGBM for all series)
        # Strategy B: Regime-Routed Engine:
        # - Smooth / Erratic -> LightGBM
        # - Intermittent / Lumpy -> IntermittentProbabilisticForecaster (Croston / NegBinomial)
        routed_preds = np.zeros_like(y_test.values)
        universal_preds = np.zeros_like(y_test.values)

        # Fit models
        for j, col in enumerate(cols):
            s_tr = y_train[col]
            regime = regimes_map[col]

            # Universal model (Seasonal Naive / LightGBM)
            u_forecaster = create_seasonal_naive(sp=7)
            u_forecaster.fit(s_tr)
            u_pred = u_forecaster.predict(fh=np.arange(1, self.horizon + 1))
            universal_preds[:, j] = np.maximum(0.0, u_pred.values.ravel())

            # Routed model
            if regime in ("Intermittent", "Lumpy"):
                inter_f = IntermittentProbabilisticForecaster()
                inter_f.fit(s_tr)
                q_inter = inter_f.predict_quantiles(horizon=self.horizon, quantiles=[0.50])
                routed_preds[:, j] = q_inter[0.50].values
            else:
                theta = create_theta_forecaster(sp=7)
                theta.fit(s_tr)
                t_pred = theta.predict(fh=np.arange(1, self.horizon + 1))
                routed_preds[:, j] = np.maximum(0.0, t_pred.values.ravel())

        regime_eval = []
        for regime_name in ["Smooth", "Intermittent", "Erratic", "Lumpy"]:
            sub_cols = [c for c, r in regimes_map.items() if r == regime_name]
            if not sub_cols:
                continue

            indices = [cols.get_loc(c) for c in sub_cols]
            act_r = y_test.values[:, indices].ravel()
            uni_r = universal_preds[:, indices].ravel()
            rout_r = routed_preds[:, indices].ravel()

            uni_wape = weighted_absolute_percentage_error(act_r, uni_r)
            rout_wape = weighted_absolute_percentage_error(act_r, rout_r)
            improvement = ((uni_wape - rout_wape) / uni_wape) * 100.0 if uni_wape > 0 else 0.0

            regime_eval.append({
                "Demand Regime": regime_name,
                "Series Count": len(sub_cols),
                "Universal Model WAPE (%)": round(uni_wape, 2),
                "Regime-Routed WAPE (%)": round(rout_wape, 2),
                "Routed Improvement (%)": round(improvement, 2),
            })

        return pd.DataFrame(regime_eval)

    # =========================================================================
    # Experiment 6: Forecast Horizon Deterioration Dynamics
    # =========================================================================
    def run_experiment_6_horizon_deterioration(self) -> pd.DataFrame:
        """Evaluates cumulative and marginal forecast deterioration across H in {7, 14, 21, 28}."""
        logger.info("Executing Experiment 6: Forecast Horizon Deterioration Dynamics...")
        s_total_train = self.all_wide["Total"].iloc[: self.train_size]
        s_total_test = self.all_wide["Total"].iloc[self.train_size : self.train_size + self.horizon]

        # Model 1: Seasonal Naive
        sn = create_seasonal_naive(sp=7)
        sn.fit(s_total_train)
        sn_preds = sn.predict(fh=np.arange(1, self.horizon + 1)).values.ravel()

        # Model 2: Recursive LightGBM
        df_tot = pd.DataFrame({"sales": s_total_train.values}, index=s_total_train.index)
        df_feat = create_lag_features(df_tot, target_col="sales", lags=[1, 7])
        df_feat = create_rolling_features(
            df_feat, target_col="sales", mean_windows=[7], std_windows=[], shift=1
        )
        df_feat["dayofweek"] = df_feat.index.dayofweek
        clean = df_feat.dropna()

        lgb_model = lgb.LGBMRegressor(n_estimators=40, learning_rate=0.08, verbose=-1, random_state=42)
        lgb_model.fit(clean.drop(columns=["sales"]), clean["sales"])

        rec_preds = []
        hist = list(s_total_train.values)
        for h in range(self.horizon):
            d_idx = self.test_dates[h]
            feat = {
                "lag_1": hist[-1],
                "lag_7": hist[-7] if len(hist) >= 7 else hist[-1],
                "rolling_mean_7": np.mean(hist[-7:]),
                "dayofweek": d_idx.dayofweek,
            }
            p = max(0.0, float(lgb_model.predict(pd.DataFrame([feat]))[0]))
            rec_preds.append(p)
            hist.append(p)

        rec_preds = np.array(rec_preds)

        decay_records = []
        for h_step in [7, 14, 21, 28]:
            act_h = s_total_test.values[:h_step]
            sn_h = sn_preds[:h_step]
            rec_h = rec_preds[:h_step]

            sn_wape = weighted_absolute_percentage_error(act_h, sn_h)
            rec_wape = weighted_absolute_percentage_error(act_h, rec_h)
            rec_rmse = root_mean_squared_error(act_h, rec_h)

            decay_records.append({
                "Horizon Window": f"1 to {h_step} Days",
                "Horizon (H)": h_step,
                "Seasonal Naive WAPE (%)": round(sn_wape, 2),
                "Recursive LightGBM WAPE (%)": round(rec_wape, 2),
                "LightGBM RMSE": round(rec_rmse, 2),
                "LightGBM Advantage (%)": round(((sn_wape - rec_wape) / sn_wape) * 100.0, 2),
            })

        return pd.DataFrame(decay_records)

