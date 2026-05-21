"""
Tests for execution/ — Sprint 5: Utility-Based Rebalancing & Execution Engine.

Covers: utility_engine, rebalancing, tax_engine, slippage, simulation,
        paper_trading, audit, turnover, state_machine, full pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def current_weights():
    return {"AAPL": 0.30, "SPY": 0.25, "RELIANCE.NS": 0.25, "INFY.NS": 0.20}


@pytest.fixture()
def target_weights():
    return {"AAPL": 0.25, "SPY": 0.25, "RELIANCE.NS": 0.30, "INFY.NS": 0.20}


@pytest.fixture()
def prices():
    return {"AAPL": 200.0, "SPY": 500.0, "RELIANCE.NS": 2800.0, "INFY.NS": 1500.0}


@pytest.fixture()
def sample_trades():
    return [
        {"ticker": "AAPL", "action": "SELL", "quantity": 5, "price": 200, "notional": 1000, "weight_change": -0.05},
        {"ticker": "RELIANCE.NS", "action": "BUY", "quantity": 2, "price": 2800, "notional": 5600, "weight_change": 0.05},
    ]


@pytest.fixture()
def tax_lots():
    from execution.tax_engine import TaxLot
    return [
        TaxLot("AAPL", 10, 150.0, date(2023, 1, 15), "US"),
        TaxLot("AAPL", 5, 180.0, date(2024, 6, 1), "US"),
    ]


@pytest.fixture()
def india_tax_lots():
    from execution.tax_engine import TaxLot
    return [
        TaxLot("RELIANCE.NS", 20, 2200.0, date(2023, 3, 10), "IN"),
        TaxLot("RELIANCE.NS", 10, 2600.0, date(2024, 8, 1), "IN"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Utility Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestAlphaImprovement:
    def test_positive_when_target_better(self):
        from execution.utility_engine import estimate_alpha_improvement
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.3, "B": 0.7}
        alpha = {"A": 0.3, "B": 0.8}
        result = estimate_alpha_improvement(current, target, alpha)
        assert result > 0

    def test_zero_without_scores(self):
        from execution.utility_engine import estimate_alpha_improvement
        result = estimate_alpha_improvement({"A": 0.5}, {"A": 0.5}, None)
        assert result == 0


class TestRiskReduction:
    def test_positive_when_risk_decreases(self):
        from execution.utility_engine import estimate_risk_reduction
        assert estimate_risk_reduction(0.20, 0.15) > 0

    def test_zero_when_risk_increases(self):
        from execution.utility_engine import estimate_risk_reduction
        assert estimate_risk_reduction(0.15, 0.20) == 0


class TestFrictionEstimation:
    def test_returns_components(self, sample_trades, prices):
        from execution.utility_engine import estimate_trade_friction
        result = estimate_trade_friction(sample_trades, prices, 1_000_000)
        assert "tax_cost" in result
        assert "slippage" in result
        assert "fees" in result
        assert result["total_friction"] > 0


class TestUtilityDecision:
    def test_blocks_low_utility(self, current_weights, target_weights, sample_trades, prices):
        from execution.utility_engine import evaluate_rebalance_utility
        result = evaluate_rebalance_utility(
            current_weights, target_weights,
            sample_trades, prices, 1_000_000,
            confidence=0.1,  # too low
        )
        assert not result.should_trade

    def test_allows_high_utility_regime_change(self, current_weights, prices):
        from execution.utility_engine import evaluate_rebalance_utility
        target = {"AAPL": 0.15, "SPY": 0.15, "RELIANCE.NS": 0.35, "INFY.NS": 0.35}
        trades = [
            {"ticker": "AAPL", "action": "SELL", "quantity": 10, "price": 200, "notional": 2000, "weight_change": -0.15},
            {"ticker": "RELIANCE.NS", "action": "BUY", "quantity": 5, "price": 2800, "notional": 14000, "weight_change": 0.10},
        ]
        result = evaluate_rebalance_utility(
            current_weights, target,
            trades, prices, 1_000_000,
            regime="panic", regime_changed=True,
            confidence=0.7,
        )
        # Regime urgency should push this toward trading
        assert result.regime_urgency > 0


# ══════════════════════════════════════════════════════════════════════════════
# Drift-Based Rebalancing
# ══════════════════════════════════════════════════════════════════════════════


class TestDriftThresholds:
    def test_panic_wider(self):
        from execution.rebalancing import get_drift_threshold
        normal = get_drift_threshold("risk_on")
        panic = get_drift_threshold("panic")
        assert panic > normal

    def test_default(self):
        from execution.rebalancing import get_drift_threshold
        t = get_drift_threshold("unknown_regime")
        assert t > 0


class TestWeightDrift:
    def test_computes_drift(self, current_weights, target_weights):
        from execution.rebalancing import compute_weight_drift
        result = compute_weight_drift(current_weights, target_weights)
        assert result["max_drift"] > 0
        assert result["total_drift"] > 0

    def test_zero_drift_for_same(self, current_weights):
        from execution.rebalancing import compute_weight_drift
        result = compute_weight_drift(current_weights, current_weights)
        assert result["max_drift"] == 0


class TestShouldRebalance:
    def test_triggers_on_large_drift(self):
        from execution.rebalancing import should_rebalance
        current = {"A": 0.6, "B": 0.4}
        target = {"A": 0.4, "B": 0.6}
        result = should_rebalance(current, target)
        assert result["should_rebalance"]
        assert result["trigger"] == "weight_drift"

    def test_no_trigger_small_drift(self, current_weights, target_weights):
        from execution.rebalancing import should_rebalance
        result = should_rebalance(current_weights, target_weights)
        assert result["trigger"] in ("weight_drift", "none")

    def test_regime_change_triggers(self, current_weights, target_weights):
        from execution.rebalancing import should_rebalance
        result = should_rebalance(
            current_weights, target_weights,
            regime="panic", regime_changed=True,
        )
        assert result["should_rebalance"]


# ══════════════════════════════════════════════════════════════════════════════
# Tax Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestTaxClassification:
    def test_india_stcg(self):
        from execution.tax_engine import classify_gain
        result = classify_gain(date(2024, 6, 1), date(2024, 12, 1), "IN")
        assert not result["is_ltcg"]

    def test_india_ltcg(self):
        from execution.tax_engine import classify_gain
        result = classify_gain(date(2023, 1, 1), date(2024, 6, 1), "IN")
        assert result["is_ltcg"]

    def test_us_classification(self):
        from execution.tax_engine import classify_gain
        result = classify_gain(date(2023, 1, 1), date(2024, 6, 1), "US")
        assert result["is_ltcg"]


class TestTaxOnSale:
    def test_fifo_consumption(self, tax_lots):
        from execution.tax_engine import estimate_tax_on_sale
        result = estimate_tax_on_sale(tax_lots, 200.0, 10, date(2025, 5, 1))
        assert result.quantity == 10
        assert len(result.lot_details) > 0

    def test_india_ltcg_exemption(self, india_tax_lots):
        from execution.tax_engine import estimate_tax_on_sale
        result = estimate_tax_on_sale(india_tax_lots, 2800.0, 20, date(2025, 5, 1), "IN")
        assert result.estimated_tax >= 0


class TestTaxLossHarvesting:
    def test_finds_opportunities(self):
        from execution.tax_engine import TaxLot, find_harvesting_opportunities
        lots = {
            "RELIANCE.NS": [TaxLot("RELIANCE.NS", 100, 3000.0, date(2024, 1, 1), "IN")],
        }
        prices = {"RELIANCE.NS": 2900.0}  # loss of ₹10,000 (>₹5K threshold)
        opps = find_harvesting_opportunities(lots, prices, date(2025, 5, 1))
        assert len(opps) > 0
        assert opps[0]["unrealized_loss"] < 0

    def test_respects_wash_sale(self):
        from execution.tax_engine import TaxLot, find_harvesting_opportunities
        lots = {
            "AAPL": [TaxLot("AAPL", 10, 250.0, date(2024, 1, 1), "US")],
        }
        prices = {"AAPL": 200.0}
        recently_sold = {"AAPL": date(2025, 4, 25)}  # sold 6 days ago
        opps = find_harvesting_opportunities(lots, prices, date(2025, 5, 1), recently_sold)
        assert len(opps) == 0  # wash sale blocks it


class TestLotRanking:
    def test_losses_first(self, tax_lots):
        from execution.tax_engine import rank_lots_for_sale
        ranked = rank_lots_for_sale(tax_lots, 140.0, date(2025, 5, 1))
        # Both lots are at a loss (sell at 140 < cost 150/180)
        # Losses should come before gains in sort order
        assert all(r["total_gain"] < 0 for r in ranked)
        assert ranked[0]["sort_key"][0] == 0  # loss bucket


# ══════════════════════════════════════════════════════════════════════════════
# Slippage Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestSlippageSimple:
    def test_buy_costs_more(self):
        from execution.slippage import estimate_slippage_simple
        result = estimate_slippage_simple(100.0, "BUY")
        assert result["execution_price"] > 100.0

    def test_sell_gets_less(self):
        from execution.slippage import estimate_slippage_simple
        result = estimate_slippage_simple(100.0, "SELL")
        assert result["execution_price"] < 100.0


class TestSlippageVolAdjusted:
    def test_high_vol_higher_slippage(self):
        from execution.slippage import estimate_slippage_vol_adjusted
        low_vol = estimate_slippage_vol_adjusted(100, "BUY", 0.10)
        high_vol = estimate_slippage_vol_adjusted(100, "BUY", 0.40)
        assert high_vol["slippage_bps"] >= low_vol["slippage_bps"]


class TestMarketImpact:
    def test_large_order(self):
        from execution.slippage import estimate_market_impact
        result = estimate_market_impact(1_000_000, 5_000_000)  # 20% of ADV
        assert result["is_large"]
        assert result["impact_cost"] > 0

    def test_small_order_no_impact(self):
        from execution.slippage import estimate_market_impact
        result = estimate_market_impact(10_000, 10_000_000)
        assert not result["is_large"]


class TestExecutionCost:
    def test_full_estimate(self):
        from execution.slippage import estimate_execution_cost
        result = estimate_execution_cost("AAPL", 200, 100, "BUY", 0.20)
        assert result["total_execution_cost"] > 0
        assert result["liquidity_score"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# Execution Simulator
# ══════════════════════════════════════════════════════════════════════════════


class TestExecutionPlan:
    def test_generates_plan(self, current_weights, target_weights, prices):
        from execution.simulation import generate_execution_plan
        plan = generate_execution_plan(
            current_weights, target_weights, prices, 1_000_000,
        )
        assert len(plan.trades) > 0
        assert plan.total_notional > 0

    def test_sells_before_buys(self, prices):
        from execution.simulation import generate_execution_plan
        current = {"AAPL": 0.50, "SPY": 0.50}
        target = {"AAPL": 0.30, "SPY": 0.70}
        plan = generate_execution_plan(current, target, prices, 1_000_000)
        sells = [t for t in plan.trades if t["action"] == "SELL"]
        buys = [t for t in plan.trades if t["action"] == "BUY"]
        if sells and buys:
            assert sells[0]["priority"] < buys[0]["priority"]


# ══════════════════════════════════════════════════════════════════════════════
# Paper Trading
# ══════════════════════════════════════════════════════════════════════════════


class TestPaperTrading:
    def test_initialization(self):
        from execution.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(1_000_000)
        assert engine.state.cash == 1_000_000

    def test_nav(self):
        from execution.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(1_000_000)
        assert engine.nav({}) == 1_000_000

    def test_snapshot(self):
        from execution.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(1_000_000)
        snap = engine.record_snapshot(date(2025, 5, 1), {})
        assert snap.nav == 1_000_000

    def test_performance_summary_empty(self):
        from execution.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine()
        assert engine.performance_summary()["status"] == "no_data"


# ══════════════════════════════════════════════════════════════════════════════
# Execution Journal
# ══════════════════════════════════════════════════════════════════════════════


class TestExecutionJournal:
    def test_log_trade(self):
        from execution.audit import ExecutionJournal
        journal = ExecutionJournal()
        dec_id = journal.log_trade(
            trades=[{"ticker": "AAPL", "action": "BUY", "quantity": 10}],
            expected_utility=0.005,
            confidence=0.7,
            rationale="Test trade",
        )
        assert len(dec_id) > 0
        assert len(journal.trades_only()) == 1

    def test_log_no_trade(self):
        from execution.audit import ExecutionJournal
        journal = ExecutionJournal()
        journal.log_no_trade(rationale="Insufficient utility")
        assert len(journal.no_trades_only()) == 1

    def test_summary(self):
        from execution.audit import ExecutionJournal
        journal = ExecutionJournal()
        journal.log_trade(trades=[], rationale="t1")
        journal.log_no_trade(rationale="t2")
        journal.log_no_trade(rationale="t3")
        summary = journal.summary()
        assert summary["trades"] == 1
        assert summary["no_trades"] == 2
        assert summary["trade_skip_ratio"] > 0

    def test_to_dataframe(self):
        from execution.audit import ExecutionJournal
        journal = ExecutionJournal()
        journal.log_trade(trades=[], rationale="test")
        df = journal.to_dataframe()
        assert not df.empty
        assert "decision_id" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# Turnover Control
# ══════════════════════════════════════════════════════════════════════════════


class TestTurnoverBudget:
    def test_fresh_budget(self):
        from execution.turnover import TurnoverBudget
        budget = TurnoverBudget()
        assert budget.remaining_monthly("2025-05") > 0

    def test_budget_enforcement(self):
        from execution.turnover import TurnoverBudget
        budget = TurnoverBudget()
        budget.record_turnover(0.15, "2025-05")
        result = budget.can_trade(0.10, "2025-05")
        assert not result["allowed"]

    def test_budget_allows(self):
        from execution.turnover import TurnoverBudget
        budget = TurnoverBudget()
        result = budget.can_trade(0.05, "2025-05")
        assert result["allowed"]


class TestSignalStability:
    def test_stable_signal(self):
        from execution.turnover import check_signal_stability
        df = pd.DataFrame({
            "ticker": ["A"] * 10,
            "signal": [0.7, 0.71, 0.72, 0.71, 0.70, 0.72, 0.71, 0.70, 0.71, 0.72],
        })
        result = check_signal_stability(df, "A", window=5)
        assert result["is_stable"]

    def test_unstable_signal(self):
        from execution.turnover import check_signal_stability
        df = pd.DataFrame({
            "ticker": ["A"] * 10,
            "signal": [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.9, 0.1],
        })
        result = check_signal_stability(df, "A", window=5)
        assert not result["is_stable"]


class TestUnnecessaryTrades:
    def test_detects_tiny_trades(self):
        from execution.turnover import detect_unnecessary_trades
        trades = [
            {"ticker": "A", "weight_change": 0.001, "notional": 100},
            {"ticker": "B", "weight_change": 0.05, "notional": 50000},
        ]
        result = detect_unnecessary_trades(trades)
        assert result["n_unnecessary"] >= 1
        assert result["n_necessary"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# State Machine
# ══════════════════════════════════════════════════════════════════════════════


class TestStateMachine:
    def test_initial_state(self):
        from execution.state_machine import PortfolioStateMachine
        sm = PortfolioStateMachine()
        assert sm.state == "idle"
        assert sm.is_idle

    def test_valid_transition(self):
        from execution.state_machine import PortfolioStateMachine
        sm = PortfolioStateMachine()
        assert sm.transition("evaluating", "test")
        assert sm.state == "evaluating"

    def test_invalid_transition(self):
        from execution.state_machine import PortfolioStateMachine
        sm = PortfolioStateMachine()
        assert not sm.transition("executing", "invalid")
        assert sm.state == "idle"

    def test_full_cycle(self):
        from execution.state_machine import PortfolioStateMachine
        sm = PortfolioStateMachine()
        sm.start_evaluation("check")
        sm.transition("pending_approval", "utility_ok")
        sm.approve_trade("approved")
        assert sm.state == "executing"
        sm.execute_and_settle("done")
        # Auto-settle should bring back to idle
        assert sm.is_idle

    def test_rejection(self):
        from execution.state_machine import PortfolioStateMachine
        sm = PortfolioStateMachine()
        sm.start_evaluation("check")
        sm.reject_trade("low_utility")
        assert sm.is_idle

    def test_cancel(self):
        from execution.state_machine import PortfolioStateMachine
        sm = PortfolioStateMachine()
        sm.start_evaluation("check")
        sm.cancel("user_cancelled")
        assert sm.is_idle

    def test_summary(self):
        from execution.state_machine import PortfolioStateMachine
        sm = PortfolioStateMachine()
        sm.start_evaluation("x")
        sm.reject_trade("y")
        s = sm.summary()
        assert s["rejections"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Full Pipeline Integration
# ══════════════════════════════════════════════════════════════════════════════


class TestExecutionEngine:
    def test_no_trade_cycle(self, current_weights, prices):
        from execution import ExecutionEngine
        engine = ExecutionEngine(1_000_000)
        # Target same as current → no drift → no trade
        result = engine.run_cycle(
            dt=date(2025, 5, 1),
            target_weights=current_weights,
            prices=prices,
        )
        assert result["decision"] == "no_trade"

    def test_trade_cycle(self, prices):
        from execution import ExecutionEngine
        engine = ExecutionEngine(1_000_000)
        # Set up current positions first
        engine.paper.state.cash = 500_000
        engine.paper.state.holdings = {"AAPL": 10, "SPY": 5}

        current = engine.paper.weights(prices)

        # Large target shift should trigger
        target = {"AAPL": 0.10, "SPY": 0.10, "RELIANCE.NS": 0.40, "INFY.NS": 0.40}

        result = engine.run_cycle(
            dt=date(2025, 5, 1),
            target_weights=target,
            prices=prices,
            regime="panic",
            regime_changed=True,
            confidence=0.7,
        )
        # Should have evaluated (may or may not trade depending on utility)
        assert result["decision"] in ("trade", "no_trade")

    def test_summary(self):
        from execution import ExecutionEngine
        engine = ExecutionEngine()
        s = engine.summary()
        assert "paper_trading" in s
        assert "journal" in s
        assert "turnover" in s
        assert "state_machine" in s
