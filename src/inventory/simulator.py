"""Discrete-event Monte Carlo inventory simulation engine for retail supply chains.

Implements:
1. Daily discrete-event replenishment cycle under continuous (s, S) and (s, Q) policies.
2. Stochastic and deterministic supplier lead times.
3. Accurate tracking of On-Hand, On-Order (Pipeline), Backorders, and Inventory Position.
4. Comprehensive supply chain KPIs: Fill Rate, Cycle Service Level, Inventory Turnover, DIO, and TCOI.
5. Working Capital vs Service Level Pareto efficient frontier evaluation.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.inventory.decision_engine import (
    InventoryDecisionEngine,
    InventoryDecisionOutput,
    InventoryPolicyParams,
)
from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Inventory.Simulator")


@dataclass
class SimulationConfig:
    """Configuration for discrete-event inventory simulation."""

    policy_type: str = "(s, S)"  # "(s, S)" or "(s, Q)"
    initial_on_hand: Optional[float] = None  # Defaults to order_up_to_level S
    lost_sales: bool = False  # If True, unmet demand is lost; if False, backordered
    stochastic_lead_time: bool = False  # If True, samples LT ~ max(1, Normal(mu_L, sigma_L))
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationKPIs:
    """Comprehensive performance metrics summarized over the simulation horizon."""

    sku_id: str
    policy_type: str
    total_days: int
    total_demand: float
    total_sales: float
    total_stockout_units: float
    fill_rate_pct: float
    cycle_service_level_pct: float
    stockout_days: int
    total_orders_placed: int
    average_on_hand: float
    average_inventory_value: float
    annualized_inventory_turnover: float
    days_of_inventory_outstanding: float
    total_holding_cost: float
    total_ordering_cost: float
    total_stockout_cost: float
    total_cost_of_inventory: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InventorySimulator:
    """Simulates daily supply chain inventory dynamics under customer demand."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initializes simulator with configuration."""
        self.config = config or SimulationConfig()
        self.rng = np.random.RandomState(self.config.random_seed)

    def _sample_lead_time(self, mu_l: float, sigma_l: float) -> int:
        """Samples integer lead time in days (minimum 1 day)."""
        if not self.config.stochastic_lead_time or sigma_l <= 0.0:
            return max(1, int(round(mu_l)))
        sample = self.rng.normal(loc=mu_l, scale=sigma_l)
        return max(1, int(round(sample)))

    def simulate_series(
        self,
        demand: pd.Series,
        decision: InventoryDecisionOutput,
        policy_params: InventoryPolicyParams,
        config: Optional[SimulationConfig] = None,
    ) -> Tuple[SimulationKPIs, pd.DataFrame]:
        """Executes daily discrete-event inventory simulation for a single SKU.

        Daily Timeline:
            1. Shipment arrivals: Orders scheduled to arrive on day t are added to on_hand.
            2. Demand fulfillment: Customer orders fulfilled from available physical stock.
            3. Replenishment review: If inventory position IP <= s, place order.
            4. Cost accounting: Holding cost on ending on_hand, ordering cost, and stockout penalties.

        Args:
            demand: Historical or forecasted daily demand series.
            decision: Target replenishment levels (s, S, EOQ, etc.).
            policy_params: Cost parameters and lead time distributions.
            config: Optional override of simulation configuration.

        Returns:
            Tuple of (SimulationKPIs, daily_records_dataframe).
        """
        cfg = config or self.config
        s = decision.reorder_point
        S = decision.order_up_to_level
        Q = decision.order_quantity

        # Initial conditions: default to mid-cycle inventory (s + Q/2) for realistic steady-state
        init_oh = cfg.initial_on_hand if cfg.initial_on_hand is not None else (s + Q / 2.0)
        on_hand = float(max(0.0, init_oh))
        on_order = 0.0
        backorders = 0.0

        # Queue of pending pipeline shipments: list of (arrival_day, qty)
        pipeline_orders: List[Tuple[int, float]] = []

        daily_holding_rate = (policy_params.unit_cost * policy_params.annual_holding_rate) / 365.0
        order_cost = policy_params.order_cost
        stockout_penalty = policy_params.stockout_penalty

        T = len(demand)
        records = []

        total_orders_placed = 0
        stockout_days_count = 0

        for t in range(T):
            date_val = str(demand.index[t]) if hasattr(demand.index, "strftime") else f"Day_{t}"
            d_t = float(max(0.0, demand.iloc[t]))
            beg_oh = on_hand

            # -----------------------------------------------------------------
            # Step 1: Receive Inbound Shipments
            # -----------------------------------------------------------------
            received_qty = 0.0
            remaining_pipeline = []
            for arr_day, qty in pipeline_orders:
                if arr_day <= t:
                    received_qty += qty
                else:
                    remaining_pipeline.append((arr_day, qty))

            pipeline_orders = remaining_pipeline
            on_hand += received_qty
            on_order = max(0.0, on_order - received_qty)
            avail_oh = on_hand

            # -----------------------------------------------------------------
            # Step 2: Demand Fulfillment
            # -----------------------------------------------------------------
            effective_demand = d_t + backorders
            fulfilled = min(effective_demand, on_hand)
            unmet = max(0.0, effective_demand - fulfilled)

            sales = fulfilled
            on_hand -= fulfilled

            if unmet > 0:
                stockout_days_count += 1
                if cfg.lost_sales:
                    lost_sales_qty = unmet
                    backorders = 0.0
                else:
                    lost_sales_qty = 0.0
                    backorders = unmet
            else:
                lost_sales_qty = 0.0
                backorders = 0.0

            # -----------------------------------------------------------------
            # Step 3: Replenishment Review
            # -----------------------------------------------------------------
            # Inventory Position = On-Hand + On-Order - Backorders
            inv_position = on_hand + on_order - backorders
            placed_qty = 0.0
            arr_step = None

            if inv_position <= s:
                if cfg.policy_type == "(s, Q)":
                    # Place order of size Q (or integer multiple if deficit is large)
                    deficit = s - inv_position
                    multiplier = max(1, int(math.ceil(deficit / max(1.0, Q))))
                    placed_qty = float(multiplier * Q)
                else:  # (s, S) policy
                    placed_qty = float(max(1.0, S - inv_position))

                lt_days = self._sample_lead_time(
                    policy_params.lead_time_mean, policy_params.lead_time_std
                )
                arr_step = t + lt_days
                pipeline_orders.append((arr_step, placed_qty))
                on_order += placed_qty
                total_orders_placed += 1

            # -----------------------------------------------------------------
            # Step 4: Cost Accounting
            # -----------------------------------------------------------------
            c_hold = on_hand * daily_holding_rate
            c_order = order_cost if placed_qty > 0 else 0.0
            c_stockout = unmet * stockout_penalty
            c_total = c_hold + c_order + c_stockout

            records.append({
                "date": date_val,
                "step": t,
                "beginning_on_hand": round(beg_oh, 2),
                "received_qty": round(received_qty, 2),
                "available_on_hand": round(avail_oh, 2),
                "demand": round(d_t, 2),
                "sales": round(sales, 2),
                "unmet_demand": round(unmet, 2),
                "backorders": round(backorders, 2),
                "ending_on_hand": round(on_hand, 2),
                "on_order": round(on_order, 2),
                "inventory_position": round(inv_position, 2),
                "placed_order_qty": round(placed_qty, 2),
                "arrival_step": arr_step,
                "holding_cost": round(c_hold, 3),
                "ordering_cost": round(c_order, 2),
                "stockout_cost": round(c_stockout, 3),
                "total_cost": round(c_total, 3),
            })

        df_sim = pd.DataFrame(records)

        # ---------------------------------------------------------------------
        # Aggregate KPI Computation
        # ---------------------------------------------------------------------
        tot_demand = float(df_sim["demand"].sum())
        tot_sales = float(df_sim["sales"].sum())
        tot_unmet = float(df_sim["unmet_demand"].sum())
        tot_stockout = tot_unmet

        fill_rate = (1.0 - (tot_stockout / tot_demand)) * 100.0 if tot_demand > 0 else 100.0
        fill_rate = max(0.0, min(100.0, fill_rate))

        csl = ((T - stockout_days_count) / T) * 100.0 if T > 0 else 100.0
        avg_oh = float(df_sim["ending_on_hand"].mean())
        avg_inv_val = avg_oh * policy_params.unit_cost

        daily_demand_rate = tot_demand / T if T > 0 else 1.0
        dio = avg_oh / daily_demand_rate if daily_demand_rate > 0 else 0.0
        turnover = (tot_sales * (365.0 / T)) / max(1e-4, avg_oh) if avg_oh > 0 else 0.0

        tot_h_cost = float(df_sim["holding_cost"].sum())
        tot_o_cost = float(df_sim["ordering_cost"].sum())
        tot_s_cost = float(df_sim["stockout_cost"].sum())
        tcoi = tot_h_cost + tot_o_cost + tot_s_cost

        kpis = SimulationKPIs(
            sku_id=decision.sku_id,
            policy_type=cfg.policy_type,
            total_days=T,
            total_demand=round(tot_demand, 1),
            total_sales=round(tot_sales, 1),
            total_stockout_units=round(tot_stockout, 1),
            fill_rate_pct=round(fill_rate, 2),
            cycle_service_level_pct=round(csl, 2),
            stockout_days=stockout_days_count,
            total_orders_placed=total_orders_placed,
            average_on_hand=round(avg_oh, 2),
            average_inventory_value=round(avg_inv_val, 2),
            annualized_inventory_turnover=round(turnover, 2),
            days_of_inventory_outstanding=round(dio, 2),
            total_holding_cost=round(tot_h_cost, 2),
            total_ordering_cost=round(tot_o_cost, 2),
            total_stockout_cost=round(tot_s_cost, 2),
            total_cost_of_inventory=round(tcoi, 2),
        )

        return kpis, df_sim

    def simulate_portfolio(
        self,
        wide_demand: pd.DataFrame,
        decisions: Dict[str, InventoryDecisionOutput],
        policy_params: InventoryPolicyParams,
        config: Optional[SimulationConfig] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Simulates inventory policies across all series in an enterprise portfolio.

        Args:
            wide_demand: DataFrame with DatetimeIndex and SKU series columns.
            decisions: Dict mapping SKU id to InventoryDecisionOutput.
            policy_params: Shared inventory policy settings.
            config: Optional simulation configuration override.

        Returns:
            Tuple of:
            - summary_kpis_df: Leaderboard DataFrame of KPIs per SKU.
            - daily_logs_dict: Dict mapping SKU id to daily trajectory DataFrame.
        """
        kpi_list = []
        logs_dict = {}

        for col in wide_demand.columns:
            s_id = str(col)
            if s_id not in decisions:
                continue

            dec = decisions[s_id]
            kpis, df_log = self.simulate_series(
                demand=wide_demand[col],
                decision=dec,
                policy_params=policy_params,
                config=config,
            )
            kpi_list.append(kpis.to_dict())
            logs_dict[s_id] = df_log

        summary_df = pd.DataFrame(kpi_list)
        return summary_df, logs_dict

    def evaluate_service_level_tradeoff(
        self,
        demand: pd.Series,
        sku_id: str,
        policy_params: InventoryPolicyParams,
        service_levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Constructs Pareto efficient frontier between Service Level and Working Capital.

        Args:
            demand: Historical demand series for simulation.
            sku_id: SKU identifier.
            policy_params: Cost parameters.
            service_levels: Sequence of target service levels in [0.70, 0.999].

        Returns:
            DataFrame tracing working capital investment, fill rate, and total cost.
        """
        sl_list = service_levels or [0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99]
        engine = InventoryDecisionEngine()

        d_mean = float(demand.mean())
        d_std = float(demand.std())

        curve_records = []
        for sl in sl_list:
            p_custom = InventoryPolicyParams(
                lead_time_mean=policy_params.lead_time_mean,
                lead_time_std=policy_params.lead_time_std,
                service_level=sl,
                unit_cost=policy_params.unit_cost,
                annual_holding_rate=policy_params.annual_holding_rate,
                order_cost=policy_params.order_cost,
                stockout_penalty=policy_params.stockout_penalty,
            )
            engine_custom = InventoryDecisionEngine(params=p_custom)
            dec = engine_custom.compute_policy(
                sku_id=sku_id, demand_mean=d_mean, demand_std=d_std
            )

            kpis, _ = self.simulate_series(
                demand=demand,
                decision=dec,
                policy_params=p_custom,
            )

            curve_records.append({
                "Target Service Level": sl,
                "Target CSL (%)": round(sl * 100.0, 1),
                "Safety Stock (Units)": dec.safety_stock,
                "Reorder Point (Units)": dec.reorder_point,
                "Achieved Fill Rate (%)": kpis.fill_rate_pct,
                "Achieved CSL (%)": kpis.cycle_service_level_pct,
                "Average Inventory Value ($)": kpis.average_inventory_value,
                "Total Cost of Inventory ($)": kpis.total_cost_of_inventory,
                "Stockout Days": kpis.stockout_days,
            })

        return pd.DataFrame(curve_records)

