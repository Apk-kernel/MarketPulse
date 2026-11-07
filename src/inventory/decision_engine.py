"""Inventory Decision Engine for Safety Stock, Reorder Point, and EOQ calculations.

Formulations:
1. Classical Gaussian Lead-Time Demand (Stochastic Demand and Stochastic Lead-Time).
2. Probabilistic Quantile Inversion (Empirical non-parametric quantiles from LightGBM/Conformal).
3. Intermittent / Slow-Mover Demand (Compound Poisson / Zero-Inflated Negative Binomial).
4. Wilson Economic Order Quantity (EOQ) and Optimal (s, S) Levels.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger("MarketPulse.Inventory.DecisionEngine")


@dataclass
class InventoryPolicyParams:
    """Parameters defining cost structures, lead times, and target service levels."""

    lead_time_mean: float = 3.0  # Average lead time in days
    lead_time_std: float = 1.0   # Standard deviation of lead time in days
    review_period: int = 1       # Days between review (1 = continuous review)
    service_level: float = 0.95  # Target service level alpha in [0.50, 0.999]
    service_level_type: str = "CSL"  # "CSL" (Cycle Service Level) or "FillRate"
    unit_cost: float = 10.0      # Purchase cost per SKU unit ($)
    annual_holding_rate: float = 0.25  # Annual inventory holding fraction (e.g. 25% of unit cost)
    order_cost: float = 25.0     # Fixed cost per purchase order ($)
    stockout_penalty: float = 15.0  # Penalty per unsatisfied demand unit ($)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InventoryDecisionOutput:
    """Actionable replenishment parameters computed by the Decision Engine."""

    sku_id: str
    safety_stock: float
    reorder_point: float
    order_quantity: float
    order_up_to_level: float
    expected_ltd: float
    sigma_ltd: float
    service_factor_z: float
    estimated_annual_demand: float
    estimated_annual_holding_cost: float
    estimated_annual_ordering_cost: float
    estimated_total_annual_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InventoryDecisionEngine:
    """Calculates optimal safety stocks, reorder points, and ordering policies."""

    def __init__(self, params: Optional[InventoryPolicyParams] = None):
        """Initializes decision engine with policy parameters."""
        self.params = params or InventoryPolicyParams()

    @staticmethod
    def calculate_ltd_distribution(
        demand_mean: float,
        demand_std: float,
        lead_time_mean: float,
        lead_time_std: float = 0.0,
    ) -> Tuple[float, float]:
        """Calculates mean and standard deviation of Lead-Time Demand (LTD).

        Mathematical Formulation (Law of Total Variance):
            E[LTD] = mu_D * mu_L
            Var(LTD) = mu_L * sigma_D^2 + mu_D^2 * sigma_L^2
            sigma_LTD = sqrt(Var(LTD))

        Args:
            demand_mean: Daily mean demand.
            demand_std: Daily demand standard deviation.
            lead_time_mean: Average supplier lead time in days.
            lead_time_std: Supplier lead time standard deviation in days.

        Returns:
            Tuple of (expected_ltd, sigma_ltd).
        """
        mu_ltd = max(0.0, demand_mean * lead_time_mean)
        var_ltd = (lead_time_mean * (demand_std ** 2)) + ((demand_mean ** 2) * (lead_time_std ** 2))
        sigma_ltd = math.sqrt(max(1e-6, var_ltd))
        return mu_ltd, sigma_ltd

    @staticmethod
    def calculate_safety_stock_gaussian(
        sigma_ltd: float,
        service_level: float = 0.95,
    ) -> Tuple[float, float]:
        """Derives Safety Stock under Gaussian lead-time demand assumption.

        Formula:
            z = Phi^(-1)(service_level)
            SS = z * sigma_LTD
        """
        alpha = max(0.50, min(0.9999, service_level))
        z = float(stats.norm.ppf(alpha))
        ss = max(0.0, z * sigma_ltd)
        return ss, z

    @staticmethod
    def calculate_eoq(
        annual_demand: float,
        order_cost: float,
        unit_cost: float,
        annual_holding_rate: float,
    ) -> float:
        """Calculates Wilson Economic Order Quantity (EOQ).

        Formula:
            h = unit_cost * annual_holding_rate
            EOQ = sqrt((2 * K * D) / h)
        """
        d = max(1.0, annual_demand)
        h = max(1e-4, unit_cost * annual_holding_rate)
        k = max(1e-4, order_cost)
        eoq = math.sqrt((2.0 * k * d) / h)
        return max(1.0, eoq)

    def calculate_safety_stock_probabilistic(
        self,
        predicted_quantiles: Dict[float, float],
        expected_ltd: float,
        service_level: float = 0.95,
    ) -> Tuple[float, float]:
        """Derives Safety Stock using non-parametric predictive quantiles.

        Directly inverts the forecasted distribution:
            ROP = Q(service_level)
            SS = max(0, ROP - E[LTD])
        """
        alpha = round(max(0.05, min(0.95, service_level)), 2)

        # Interpolate if exact quantile not in dict
        avail_q = sorted(predicted_quantiles.keys())
        if alpha in predicted_quantiles:
            rop = predicted_quantiles[alpha]
        else:
            q_vals = [predicted_quantiles[q] for q in avail_q]
            rop = float(np.interp(alpha, avail_q, q_vals))

        ss = max(0.0, rop - expected_ltd)
        z = float(stats.norm.ppf(alpha))
        return ss, z

    def calculate_safety_stock_intermittent(
        self,
        occurrence_prob: float,
        size_mean: float,
        size_var: float,
        lead_time_mean: float,
        service_level: float = 0.95,
    ) -> Tuple[float, float]:
        """Derives Safety Stock for intermittent SKUs using compound lead-time demand.

        Compounded over lead time L:
            Expected positive demand days: n_L = L * pi
            Total expected LTD: mu_LTD = L * pi * mu_z
            Var(LTD) = L * pi * sigma_z^2 + L * pi * (1 - pi) * mu_z^2
        """
        pi = max(1e-4, min(1.0, occurrence_prob))
        mu_z = max(1e-4, size_mean)
        var_z = max(1e-4, size_var)
        L = max(1.0, lead_time_mean)

        mu_ltd = L * pi * mu_z
        var_ltd = (L * pi * var_z) + (L * pi * (1.0 - pi) * (mu_z ** 2))
        sigma_ltd = math.sqrt(max(1e-4, var_ltd))

        # Quantile derivation
        if var_ltd > 1.05 * mu_ltd:
            p_nb = max(1e-4, min(0.999, mu_ltd / var_ltd))
            n_nb = max(1e-4, (mu_ltd ** 2) / (var_ltd - mu_ltd))
            rop = float(stats.nbinom.ppf(service_level, n=n_nb, p=p_nb))
        else:
            rop = float(stats.poisson.ppf(service_level, mu=mu_ltd))

        ss = max(0.0, rop - mu_ltd)
        z = float(stats.norm.ppf(service_level))
        return ss, z

    def compute_policy(
        self,
        sku_id: str,
        demand_mean: float,
        demand_std: float,
        predicted_quantiles: Optional[Dict[float, float]] = None,
        demand_profile: Optional[Any] = None,
    ) -> InventoryDecisionOutput:
        """Computes comprehensive inventory policy parameters for a specific SKU.

        Args:
            sku_id: Unique SKU identifier.
            demand_mean: Average daily demand rate.
            demand_std: Standard deviation of daily demand.
            predicted_quantiles: Optional dictionary of predictive quantiles.
            demand_profile: Optional DemandProfile object from demand diagnostics.

        Returns:
            InventoryDecisionOutput containing SS, ROP, EOQ, S, and cost estimates.
        """
        p = self.params
        mu_ltd, sigma_ltd = self.calculate_ltd_distribution(
            demand_mean=demand_mean,
            demand_std=demand_std,
            lead_time_mean=p.lead_time_mean,
            lead_time_std=p.lead_time_std,
        )

        # Determine Safety Stock based on available intelligence
        if demand_profile is not None and getattr(demand_profile, "regime", None) in ("Intermittent", "Lumpy"):
            # Intermittent compound distribution
            occ_prob = 1.0 - getattr(demand_profile, "zero_ratio", 0.5)
            nz_mean = getattr(demand_profile, "non_zero_mean", demand_mean)
            nz_std = getattr(demand_profile, "non_zero_std", demand_std)
            ss, z = self.calculate_safety_stock_intermittent(
                occurrence_prob=occ_prob,
                size_mean=nz_mean,
                size_var=nz_std ** 2,
                lead_time_mean=p.lead_time_mean,
                service_level=p.service_level,
            )
        elif predicted_quantiles is not None and len(predicted_quantiles) > 0:
            # Empirical quantile inversion
            ss, z = self.calculate_safety_stock_probabilistic(
                predicted_quantiles=predicted_quantiles,
                expected_ltd=mu_ltd,
                service_level=p.service_level,
            )
        else:
            # Gaussian formula
            ss, z = self.calculate_safety_stock_gaussian(
                sigma_ltd=sigma_ltd,
                service_level=p.service_level,
            )

        rop = mu_ltd + ss

        # EOQ and Order-Up-To Level (S)
        annual_demand = demand_mean * 365.0
        eoq = self.calculate_eoq(
            annual_demand=annual_demand,
            order_cost=p.order_cost,
            unit_cost=p.unit_cost,
            annual_holding_rate=p.annual_holding_rate,
        )

        order_up_to_s = rop + eoq

        # Annualized expected cost accounting
        annual_h_cost = (ss + (eoq / 2.0)) * (p.unit_cost * p.annual_holding_rate)
        annual_o_cost = (annual_demand / max(1.0, eoq)) * p.order_cost
        total_cost = annual_h_cost + annual_o_cost

        return InventoryDecisionOutput(
            sku_id=sku_id,
            safety_stock=round(ss, 2),
            reorder_point=round(rop, 2),
            order_quantity=round(eoq, 2),
            order_up_to_level=round(order_up_to_s, 2),
            expected_ltd=round(mu_ltd, 2),
            sigma_ltd=round(sigma_ltd, 2),
            service_factor_z=round(z, 3),
            estimated_annual_demand=round(annual_demand, 1),
            estimated_annual_holding_cost=round(annual_h_cost, 2),
            estimated_annual_ordering_cost=round(annual_o_cost, 2),
            estimated_total_annual_cost=round(total_cost, 2),
        )

