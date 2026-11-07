"""Publication-quality Exploratory Data Analysis (EDA) and reporting generator."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Configure matplotlib cache and headless backend
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "marketpulse_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.hierarchy import M5Hierarchy
from src.features.demand_features import classify_demand_regimes
from src.features.temporal_features import (
    calculate_event_uplift,
    calculate_seasonal_indices,
    extract_calendar_features,
)
from src.features.price_features import calculate_price_features, estimate_price_elasticity
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.EDAReporter")

# Set publication styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


class EDAReporter:
    """Orchestrates comprehensive exploratory time-series and demand analysis."""

    def __init__(self, output_dir: Path):
        """Initializes reporter with target output directory.

        Args:
            output_dir: Path to directory where figures and reports will be saved.
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_sales_distribution(self, df: pd.DataFrame, sales_col: str = "sales") -> Path:
        """Plots sales distributions: raw sales histogram, zero vs non-zero ratio, and log1p density."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        sales = df[sales_col].dropna()
        non_zero = sales[sales > 0]
        zero_pct = (len(sales) - len(non_zero)) / len(sales) * 100 if len(sales) else 0

        # Subplot 1: Zero vs Non-Zero Count
        axes[0].bar(["Zero Demand", "Positive Sales"], [len(sales) - len(non_zero), len(non_zero)], color=["#e74c3c", "#2ecc71"])
        axes[0].set_title(f"Demand Sparsity ({zero_pct:.1f}% Zero Sales)")
        axes[0].set_ylabel("Observation Count")
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        # Subplot 2: Non-Zero Demand Distribution (Truncated 99th percentile)
        p99 = np.percentile(non_zero, 99) if len(non_zero) else 10
        sns.histplot(non_zero[non_zero <= p99], bins=30, kde=True, ax=axes[1], color="#3498db")
        axes[1].set_title(f"Non-Zero Demand Distribution (<= 99th pct: {p99:.0f})")
        axes[1].set_xlabel("Units Sold")
        axes[1].set_ylabel("Frequency")

        # Subplot 3: Log-transformed demand density
        log_sales = np.log1p(sales)
        sns.kdeplot(log_sales, ax=axes[2], fill=True, color="#9b59b6")
        axes[2].set_title("Log-Transformed Demand: ln(Sales + 1)")
        axes[2].set_xlabel("ln(Sales + 1)")
        axes[2].set_ylabel("Density")

        plt.suptitle("Analysis A: Sales & Demand Sparsity Distribution", y=1.02)
        plt.tight_layout()
        out_path = self.figures_dir / "01_sales_distribution.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_temporal_behaviour(self, df: pd.DataFrame, sales_col: str = "sales") -> Path:
        """Plots temporal patterns: Daily overall trend, Day-of-week profile, and Monthly seasonality."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Daily aggregate demand trend
        daily = df.groupby("date", observed=True)[sales_col].sum().reset_index()
        axes[0].plot(daily["date"], daily[sales_col], color="#2980b9", lw=1.5)
        # 7-day rolling average
        rolling_7 = daily[sales_col].rolling(7, min_periods=1).mean()
        axes[0].plot(daily["date"], rolling_7, color="#e67e22", lw=2, label="7-Day Rolling Mean")
        axes[0].set_title("Total Daily Demand Trend")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Units Sold")
        axes[0].legend()
        axes[0].tick_params(axis="x", rotation=30)

        # Subplot 2: Day of Week Seasonality
        dow_indices = calculate_seasonal_indices(df, sales_col=sales_col)["dow_profile"]
        bars = axes[1].bar(dow_indices["day_name"], dow_indices["seasonal_index"], color="#1abc9c")
        axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (1.0)")
        axes[1].set_title("Day-of-Week Seasonality Index")
        axes[1].set_ylabel("Seasonal Index (Mean / Overall Mean)")
        axes[1].tick_params(axis="x", rotation=40)
        axes[1].legend()

        # Subplot 3: Monthly Seasonality
        month_indices = calculate_seasonal_indices(df, sales_col=sales_col)["monthly_profile"]
        axes[2].plot(month_indices["month"], month_indices["seasonal_index"], marker="o", color="#8e44ad", lw=2)
        axes[2].axhline(1.0, color="gray", linestyle="--", alpha=0.7)
        axes[2].set_title("Monthly Seasonality Index")
        axes[2].set_xlabel("Month")
        axes[2].set_ylabel("Seasonal Index")
        axes[2].set_xticks(range(1, 13))

        plt.suptitle("Analysis B: Temporal Dynamics & Seasonality Structure", y=1.02)
        plt.tight_layout()
        out_path = self.figures_dir / "02_temporal_behaviour.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_demand_regimes_scatter(self, regimes_df: pd.DataFrame) -> Path:
        """Plots Syntetos-Boylan demand regime quadrant scatter (ADI vs CV^2)."""
        fig, ax = plt.subplots(figsize=(10, 7))

        regime_colors = {
            "Smooth": "#2ecc71",
            "Intermittent": "#f1c40f",
            "Erratic": "#e67e22",
            "Lumpy": "#e74c3c",
            "ZeroDemand": "#95a5a6",
        }

        for reg, color in regime_colors.items():
            sub = regimes_df[regimes_df["regime"] == reg]
            if len(sub) > 0:
                ax.scatter(sub["adi"], sub["cv2"], label=f"{reg} (n={len(sub)})", color=color, alpha=0.7, edgecolors="none", s=50)

        # Draw classification boundaries
        ax.axvline(1.32, color="black", linestyle="--", lw=1.2, label="ADI Cutoff = 1.32")
        ax.axhline(0.49, color="black", linestyle=":", lw=1.2, label="CV^2 Cutoff = 0.49")

        ax.set_title("Analysis C: Syntetos-Boylan Demand Categorization (ADI vs CV²)")
        ax.set_xlabel("Average Demand Interval (ADI) [Periods between non-zero demand]")
        ax.set_ylabel("Squared Coefficient of Variation (CV²)")
        ax.set_xlim(0.8, max(3.5, regimes_df["adi"].quantile(0.95)))
        ax.set_ylim(-0.1, max(2.5, regimes_df["cv2"].quantile(0.95)))

        # Quadrant labels
        ax.text(1.05, 0.20, "SMOOTH\n(Regular, Low Var)", ha="center", color="#27ae60", fontweight="bold")
        ax.text(2.20, 0.20, "INTERMITTENT\n(Sporadic, Low Var)", ha="center", color="#d4ac0d", fontweight="bold")
        ax.text(1.05, 1.50, "ERRATIC\n(Regular, High Var)", ha="center", color="#d35400", fontweight="bold")
        ax.text(2.20, 1.50, "LUMPY\n(Sporadic, High Var)", ha="center", color="#c0392b", fontweight="bold")

        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        out_path = self.figures_dir / "03_demand_regimes_scatter.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_price_behaviour(self, df: pd.DataFrame, elasticity_df: pd.DataFrame) -> Path:
        """Plots price distribution, discount depth vs volume, and estimated elasticity curves."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        df_price = calculate_price_features(df)

        # Subplot 1: Price distribution by category
        if "cat_id" in df_price.columns:
            sns.boxplot(data=df_price, x="cat_id", y="sell_price", ax=axes[0], hue="cat_id", palette="Set2", legend=False)
            axes[0].set_title("Sell Price Distribution by Category")
            axes[0].set_ylabel("Sell Price ($)")
        else:
            sns.histplot(df_price["sell_price"].dropna(), ax=axes[0], kde=True, color="#34495e")
            axes[0].set_title("Overall Sell Price Distribution")

        # Subplot 2: Discount Depth vs Sales Lift
        if "discount_depth" in df_price.columns:
            df_price["discount_bin"] = pd.cut(
                df_price["discount_depth"],
                bins=[-0.01, 0.05, 0.15, 0.25, 1.0],
                labels=["Regular (<=5%)", "Light (5-15%)", "Moderate (15-25%)", "Deep (>25%)"],
            )
            disc_agg = df_price.groupby("discount_bin", observed=True)["sales"].agg(["mean", "std"]).reset_index()
            axes[1].bar(disc_agg["discount_bin"].astype(str), disc_agg["mean"], color="#e74c3c", alpha=0.8)
            axes[1].set_title("Average Daily Sales by Discount Depth")
            axes[1].set_ylabel("Average Units Sold")
            axes[1].tick_params(axis="x", rotation=25)

        # Subplot 3: Price Elasticity Summary Bar
        if not elasticity_df.empty:
            colors = ["#2ecc71" if e < -1.0 else "#e67e22" for e in elasticity_df["elasticity"]]
            axes[2].barh(elasticity_df["group"], elasticity_df["elasticity"], color=colors)
            axes[2].axvline(-1.0, color="red", linestyle="--", label="Unit Elasticity (-1.0)")
            axes[2].set_title("Estimated Price Elasticity (log Q / log P)")
            axes[2].set_xlabel("Elasticity Coefficient (Beta)")
            axes[2].legend()

        plt.suptitle("Analysis D: Price Behavior & Demand Responsiveness", y=1.02)
        plt.tight_layout()
        out_path = self.figures_dir / "04_price_behaviour.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_hierarchical_aggregations(self, df: pd.DataFrame, hierarchy: M5Hierarchy) -> Path:
        """Plots demand curves across 4 levels of the M5 hierarchy: Total, State, Store, and Category."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Level 1: Total
        lvl1 = hierarchy.aggregate_bottom_up(df, level=1)
        axes[0, 0].plot(lvl1["date"], lvl1["sales"], color="#2c3e50", lw=2)
        axes[0, 0].set_title("Hierarchy Level 1: Total Business Demand")
        axes[0, 0].set_ylabel("Total Units")

        # Level 2: State
        lvl2 = hierarchy.aggregate_bottom_up(df, level=2)
        for st in lvl2["series_id"].unique():
            sub = lvl2[lvl2["series_id"] == st]
            axes[0, 1].plot(sub["date"], sub["sales"], label=st, lw=1.5)
        axes[0, 1].set_title("Hierarchy Level 2: State-Level Demand")
        axes[0, 1].legend()

        # Level 3: Store
        lvl3 = hierarchy.aggregate_bottom_up(df, level=3)
        for store in lvl3["series_id"].unique()[:5]:  # Top 5 stores
            sub = lvl3[lvl3["series_id"] == store]
            axes[1, 0].plot(sub["date"], sub["sales"], label=store, lw=1.2)
        axes[1, 0].set_title("Hierarchy Level 3: Store-Level Demand (Sample)")
        axes[1, 0].legend()

        # Level 4: Category
        lvl4 = hierarchy.aggregate_bottom_up(df, level=4)
        for cat in lvl4["series_id"].unique():
            sub = lvl4[lvl4["series_id"] == cat]
            axes[1, 1].plot(sub["date"], sub["sales"], label=cat, lw=1.5)
        axes[1, 1].set_title("Hierarchy Level 4: Category-Level Demand")
        axes[1, 1].legend()

        for ax in axes.flat:
            ax.tick_params(axis="x", rotation=25)

        plt.suptitle("Analysis E: Hierarchical Aggregation Profiles", y=1.01)
        plt.tight_layout()
        out_path = self.figures_dir / "05_hierarchical_aggregations.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def run_full_eda(
        self,
        df: pd.DataFrame,
        hierarchy: M5Hierarchy,
        sales_col: str = "sales",
    ) -> Dict[str, Any]:
        """Executes full Phase 2 exploratory data analysis, saves publication figures, and outputs summary report.

        Args:
            df: Merged long-format demand DataFrame.
            hierarchy: M5Hierarchy instance.
            sales_col: Sales column name.

        Returns:
            Dictionary containing structured summary statistics across all five analytical areas.
        """
        logger.info("Executing full exploratory demand analysis suite...")
        summary: Dict[str, Any] = {}

        # A. Sales Distribution
        sales = df[sales_col].dropna()
        non_zero = sales[sales > 0]
        summary["A_sales_distribution"] = {
            "total_observations": int(len(sales)),
            "mean_sales": round(float(sales.mean()), 4),
            "median_sales": round(float(sales.median()), 4),
            "std_sales": round(float(sales.std()), 4),
            "zero_sales_count": int((sales == 0).sum()),
            "zero_sales_pct": round(float((sales == 0).mean() * 100), 2),
            "non_zero_mean": round(float(non_zero.mean()), 4) if len(non_zero) else 0.0,
            "non_zero_std": round(float(non_zero.std()), 4) if len(non_zero) else 0.0,
            "skewness": round(float(sales.skew()), 4),
        }
        fig_a = self.plot_sales_distribution(df, sales_col=sales_col)

        # B. Temporal Behaviour
        temporal_indices = calculate_seasonal_indices(df, sales_col=sales_col)
        event_uplift = calculate_event_uplift(df, sales_col=sales_col)
        summary["B_temporal_behaviour"] = {
            "day_of_week_seasonal_indices": temporal_indices["dow_profile"].to_dict(orient="records"),
            "monthly_seasonal_indices": temporal_indices["monthly_profile"].to_dict(orient="records"),
            "snap_uplift": event_uplift["snap"],
            "event_uplift": (
                event_uplift["events"].to_dict(orient="records")
                if not event_uplift["events"].empty
                else []
            ),
        }
        fig_b = self.plot_temporal_behaviour(df, sales_col=sales_col)

        # C. Product Demand Regimes (Syntetos-Boylan)
        regimes_df = classify_demand_regimes(df, sales_col=sales_col)
        regime_counts = regimes_df["regime"].value_counts().to_dict()
        summary["C_product_behaviour"] = {
            "total_series": len(regimes_df),
            "regime_distribution": regime_counts,
            "regime_pct": {k: round(v / len(regimes_df) * 100, 2) for k, v in regime_counts.items()},
            "avg_adi": round(float(regimes_df["adi"].mean()), 4),
            "avg_cv2": round(float(regimes_df["cv2"].mean()), 4),
            "top_demand_skus": regimes_df.nlargest(5, "mean_sales")[["series_id", "mean_sales", "regime"]].to_dict(orient="records"),
            "most_volatile_skus": regimes_df.nlargest(5, "cv2")[["series_id", "cv2", "regime"]].to_dict(orient="records"),
        }
        fig_c = self.plot_demand_regimes_scatter(regimes_df)

        # D. Price Behaviour
        df_price = calculate_price_features(df)
        elasticity_df = estimate_price_elasticity(df_price, sales_col=sales_col)
        summary["D_price_behaviour"] = {
            "avg_sell_price": round(float(df["sell_price"].mean()), 2) if "sell_price" in df.columns else None,
            "min_sell_price": round(float(df["sell_price"].min()), 2) if "sell_price" in df.columns else None,
            "max_sell_price": round(float(df["sell_price"].max()), 2) if "sell_price" in df.columns else None,
            "price_elasticity_by_category": elasticity_df.to_dict(orient="records"),
        }
        fig_d = self.plot_price_behaviour(df, elasticity_df)

        # E. Hierarchical Demand Concentration
        lvl1 = hierarchy.aggregate_bottom_up(df, level=1, sales_col=sales_col)
        lvl2 = hierarchy.aggregate_bottom_up(df, level=2, sales_col=sales_col)
        lvl4 = hierarchy.aggregate_bottom_up(df, level=4, sales_col=sales_col)
        state_share = (
            lvl2.groupby("series_id", observed=True)[sales_col].sum()
            / lvl1[sales_col].sum()
            * 100
        ).round(2).to_dict()
        cat_share = (
            lvl4.groupby("series_id", observed=True)[sales_col].sum()
            / lvl1[sales_col].sum()
            * 100
        ).round(2).to_dict()
        summary["E_hierarchical_analysis"] = {
            "total_business_demand": float(lvl1[sales_col].sum()),
            "state_volume_share_pct": state_share,
            "category_volume_share_pct": cat_share,
        }
        fig_e = self.plot_hierarchical_aggregations(df, hierarchy)

        # Save summary JSON
        summary_json_path = self.output_dir / "eda_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved EDA Summary JSON to {summary_json_path}")

        # Save Markdown Summary
        md_path = self.output_dir / "eda_summary.md"
        self._write_markdown_summary(summary, md_path)
        logger.info(f"Saved EDA Summary Markdown to {md_path}")

        return summary

    def _write_markdown_summary(self, summary: Dict[str, Any], path: Path) -> None:
        """Writes structured publication-grade EDA markdown report."""
        a = summary["A_sales_distribution"]
        b = summary["B_temporal_behaviour"]
        c = summary["C_product_behaviour"]
        d = summary["D_price_behaviour"]
        e = summary["E_hierarchical_analysis"]

        md = [
            "# MarketPulse: Comprehensive Exploratory Data Analysis Report",
            "",
            "## 1. Sales Distribution Diagnostics (Dimension A)",
            f"- **Mean Demand**: {a['mean_sales']:.2f} units/day",
            f"- **Median Demand**: {a['median_sales']:.2f} units/day",
            f"- **Standard Deviation**: {a['std_sales']:.2f}",
            f"- **Zero-Sales Frequency**: **{a['zero_sales_pct']:.2f}%** ({a['zero_sales_count']:,} periods)",
            f"- **Non-Zero Mean Demand**: {a['non_zero_mean']:.2f} units/day (Std: {a['non_zero_std']:.2f})",
            f"- **Demand Skewness**: {a['skewness']:.2f} (positive long-tailed distribution)",
            "",
            "## 2. Temporal Dynamics & Seasonality (Dimension B)",
            "### Day-of-Week Seasonality:",
        ]
        for dow in b["day_of_week_seasonal_indices"]:
            md.append(f"- **{dow['day_name']}**: Index = `{dow['seasonal_index']:.3f}`")

        snap = b.get("snap_uplift", {})
        if snap:
            md.extend([
                "",
                "### SNAP & Promotional Uplift:",
                f"- **SNAP Benefit Days Uplift**: `{snap.get('snap_uplift_pct', 0.0):+.2f}%` "
                f"(Mean SNAP sales: {snap.get('snap_on_mean', 0.0):.2f} vs Non-SNAP: {snap.get('snap_off_mean', 0.0):.2f})",
            ])

        md.extend([
            "",
            "## 3. Product Demand Regimes: Syntetos-Boylan Framework (Dimension C)",
            f"- **Total Product-Store Series Analyzed**: {c['total_series']:,}",
            f"- **Average ADI**: {c['avg_adi']:.2f} periods | **Average CV²**: {c['avg_cv2']:.2f}",
            "### Regime Distribution:",
        ])
        for reg, count in c["regime_distribution"].items():
            pct = c["regime_pct"].get(reg, 0.0)
            md.append(f"- **{reg}**: `{count}` series ({pct:.1f}%)")

        md.extend([
            "",
            "## 4. Price Elasticity & Promotional Behavior (Dimension D)",
            f"- **Price Range**: ${d.get('min_sell_price', 0):.2f} - ${d.get('max_sell_price', 0):.2f} (Average: ${d.get('avg_sell_price', 0):.2f})",
            "### Empirical Elasticity by Category (log Q / log P):",
        ])
        for el in d["price_elasticity_by_category"]:
            md.append(f"- **{el['group']}**: Elasticity = `{el['elasticity']}` ($R^2 = {el['r_squared']}$) -> *{el['interpretation']}*")

        md.extend([
            "",
            "## 5. Hierarchical Demand Concentration (Dimension E)",
            f"- **Total Business Volume**: {e['total_business_demand']:,} units",
            "### State-Level Volume Share:",
        ])
        for st, pct in e["state_volume_share_pct"].items():
            md.append(f"- **State {st}**: `{pct:.1f}%`")

        md.append("### Category-Level Volume Share:")
        for cat, pct in e["category_volume_share_pct"].items():
            md.append(f"- **Category {cat}**: `{pct:.1f}%`")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

