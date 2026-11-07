"""Unit and integration test suite for Inventory Decision Engine and Simulation."""

import math
import pytest
import numpy as np
import pandas as pd

from src.inventory.decision_engine import (
    InventoryDecisionEngine,
    InventoryDecisionOutput,
    InventoryPolicyParams,
)
from src.inventory.simulator import (
    InventorySimulator,
    SimulationConfig,
    SimulationKPIs,
)


@pytest.fixture
def policy_params():
    """Default test inventory policy parameters."""
    return InventoryPolicyParams(
        lead_time_mean=3.0,
        lead_time_std=1.0,
        service_level=0.95,
        unit_cost=10.0,
        annual_holding_rate=0.25,
        order_cost=20.0,
        stockout_penalty=15.0,
    )


@pytest.fixture
def synthetic_demand():
    """Generates a synthetic demand series for simulation tests."""
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    rng = np.random.RandomState(42)
    demand = rng.poisson(lam=5.0, size=60).astype(float)
    return pd.Series(demand, index=dates, name="test_sku")


def test_ltd_distribution():
    """Verifies law of total variance for lead-time demand."""
    mu_d, sigma_d = 10.0, 2.0
    mu_l, sigma_l = 4.0, 1.0

    mu_ltd, sigma_ltd = InventoryDecisionEngine.calculate_ltd_distribution(
        demand_mean=mu_d,
        demand_std=sigma_d,
        lead_time_mean=mu_l,
        lead_time_std=sigma_l,
    )

    expected_mu = 10.0 * 4.0
    expected_var = (4.0 * (2.0 ** 2)) + ((10.0 ** 2) * (1.0 ** 2))
    expected_sigma = math.sqrt(expected_var)

    assert mu_ltd == expected_mu
    assert pytest.approx(sigma_ltd, rel=1e-4) == expected_sigma


def test_safety_stock_gaussian():
    """Verifies standard normal critical value application in safety stock."""
    sigma_ltd = 10.0
    ss, z = InventoryDecisionEngine.calculate_safety_stock_gaussian(
        sigma_ltd=sigma_ltd, service_level=0.95
    )

    assert pytest.approx(z, abs=0.01) == 1.645
    assert pytest.approx(ss, abs=0.1) == 16.45


def test_eoq_calculation():
    """Verifies Wilson EOQ formula."""
    annual_demand = 3650.0  # 10 units/day * 365
    order_cost = 25.0
    unit_cost = 10.0
    holding_rate = 0.20  # h = 2.0 $/unit/yr

    eoq = InventoryDecisionEngine.calculate_eoq(
        annual_demand=annual_demand,
        order_cost=order_cost,
        unit_cost=unit_cost,
        annual_holding_rate=holding_rate,
    )

    # EOQ = sqrt(2 * 25 * 3650 / 2) = sqrt(91250) approx 302.076
    expected_eoq = math.sqrt((2.0 * 25.0 * 3650.0) / 2.0)
    assert pytest.approx(eoq, rel=1e-3) == expected_eoq


def test_decision_engine_compute_policy(policy_params):
    """Verifies comprehensive policy derivation output."""
    engine = InventoryDecisionEngine(params=policy_params)
    decision = engine.compute_policy(
        sku_id="SKU_001",
        demand_mean=5.0,
        demand_std=2.0,
    )

    assert isinstance(decision, InventoryDecisionOutput)
    assert decision.sku_id == "SKU_001"
    assert decision.safety_stock > 0
    assert decision.reorder_point > decision.safety_stock
    assert decision.order_quantity > 0
    assert decision.order_up_to_level > decision.reorder_point
    assert decision.estimated_total_annual_cost > 0


def test_simulation_conservation_of_mass(synthetic_demand, policy_params):
    """Verifies physical inventory conservation: Beg_OH + Received == End_OH + Sales."""
    engine = InventoryDecisionEngine(params=policy_params)
    decision = engine.compute_policy(
        sku_id="test_sku",
        demand_mean=float(synthetic_demand.mean()),
        demand_std=float(synthetic_demand.std()),
    )

    simulator = InventorySimulator(
        config=SimulationConfig(policy_type="(s, S)", stochastic_lead_time=False)
    )
    kpis, df_log = simulator.simulate_series(
        demand=synthetic_demand,
        decision=decision,
        policy_params=policy_params,
    )

    for _, row in df_log.iterrows():
        inflow = row["beginning_on_hand"] + row["received_qty"]
        outflow = row["ending_on_hand"] + row["sales"]
        assert pytest.approx(inflow, abs=1e-4) == outflow


def test_simulation_policies_comparison(synthetic_demand, policy_params):
    """Verifies that both (s, S) and (s, Q) policies execute and produce valid KPIs."""
    engine = InventoryDecisionEngine(params=policy_params)
    decision = engine.compute_policy(
        sku_id="test_sku",
        demand_mean=float(synthetic_demand.mean()),
        demand_std=float(synthetic_demand.std()),
    )

    # (s, S)
    sim_sS = InventorySimulator(config=SimulationConfig(policy_type="(s, S)"))
    kpis_sS, _ = sim_sS.simulate_series(synthetic_demand, decision, policy_params)

    # (s, Q)
    sim_sQ = InventorySimulator(config=SimulationConfig(policy_type="(s, Q)"))
    kpis_sQ, _ = sim_sQ.simulate_series(synthetic_demand, decision, policy_params)

    for kpi in [kpis_sS, kpis_sQ]:
        assert isinstance(kpi, SimulationKPIs)
        assert 0.0 <= kpi.fill_rate_pct <= 100.0
        assert 0.0 <= kpi.cycle_service_level_pct <= 100.0
        assert kpi.total_cost_of_inventory > 0.0
        assert kpi.total_sales <= kpi.total_demand


def test_pareto_frontier_monotonicity(synthetic_demand, policy_params):
    """Verifies that safety stock increases monotonically with higher service level targets."""
    simulator = InventorySimulator()
    frontier_df = simulator.evaluate_service_level_tradeoff(
        demand=synthetic_demand,
        sku_id="test_sku",
        policy_params=policy_params,
        service_levels=[0.80, 0.90, 0.95, 0.99],
    )

    assert len(frontier_df) == 4
    ss_values = frontier_df["Safety Stock (Units)"].tolist()
    assert ss_values == sorted(ss_values)

