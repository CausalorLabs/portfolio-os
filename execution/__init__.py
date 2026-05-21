"""
Execution Engine — Pipeline Orchestrator.

Sprint 5: Utility-Based Rebalancing & Execution Engine.

The system becomes operationally realistic:
  1. Evaluate: should we trade at all? (utility gating)
  2. Check drift thresholds (event-driven, not calendar)
  3. Estimate friction (tax + slippage + fees)
  4. Generate execution plan
  5. Simulate execution
  6. Update paper portfolio
  7. Log everything

The best trade is often: no trade.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from omegaconf import OmegaConf

from execution.utility_engine import evaluate_rebalance_utility
from execution.rebalancing import should_rebalance
from execution.simulation import generate_execution_plan, simulate_execution
from execution.paper_trading import PaperTradingEngine
from execution.audit import ExecutionJournal
from execution.turnover import TurnoverBudget, compute_turnover, detect_unnecessary_trades
from execution.state_machine import PortfolioStateMachine


def _load_config() -> dict:
    cfg_path = Path("configs/execution_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


class ExecutionEngine:
    """
    Full execution pipeline.

    Integrates utility gating, drift detection, turnover control,
    paper trading, and audit logging into a single operational loop.
    """

    def __init__(self, initial_capital: float | None = None):
        self.paper = PaperTradingEngine(initial_capital)
        self.journal = ExecutionJournal()
        self.turnover_budget = TurnoverBudget()
        self.state_machine = PortfolioStateMachine()

        logger.info("Execution Engine initialized")

    def run_cycle(
        self,
        dt: date,
        target_weights: dict[str, float],
        prices: dict[str, float],
        alpha_scores: dict[str, float] | None = None,
        current_vol: float = 0.0,
        target_vol: float = 0.0,
        current_dr: float = 1.0,
        target_dr: float = 1.0,
        regime: str = "risk_on",
        regime_changed: bool = False,
        confidence: float = 0.5,
        prev_confidence: float | None = None,
        current_risk_pct: dict[str, float] | None = None,
        target_risk_pct: dict[str, float] | None = None,
        country_map: dict[str, str] | None = None,
        ticker_vols: dict[str, float] | None = None,
    ) -> dict:
        """
        Run one execution cycle.

        Steps:
          1. State → evaluating
          2. Check drift thresholds
          3. Evaluate utility
          4. Check turnover budget
          5. Generate and filter execution plan
          6. Execute or skip
          7. Record snapshot
          8. Log decision

        Returns: full cycle result dict
        """
        logger.info(f"{'=' * 50}")
        logger.info(f"EXECUTION CYCLE — {dt}")
        logger.info(f"{'=' * 50}")

        result = {
            "date": dt,
            "regime": regime,
            "decision": "no_trade",
            "reason": "",
        }

        # Step 1: Start evaluation
        self.state_machine.start_evaluation(reason="daily_check")

        # Current weights
        current_weights = self.paper.weights(prices)

        # Step 2: Check drift
        logger.info("[1/5] Checking drift thresholds...")
        drift_result = should_rebalance(
            current_weights=current_weights,
            target_weights=target_weights,
            regime=regime,
            regime_changed=regime_changed,
            current_risk_pct=current_risk_pct,
            target_risk_pct=target_risk_pct,
            confidence=confidence,
            prev_confidence=prev_confidence,
        )

        if not drift_result["should_rebalance"]:
            logger.info(f"  No drift trigger — skipping")
            self.state_machine.reject_trade("no_drift_trigger")

            self.journal.log_no_trade(
                regime=regime,
                trigger="none",
                rationale=f"No drift trigger (max_drift={drift_result['drift']['max_drift']:.3f})",
            )

            self.paper.log_decision(dt, "no_trade", drift_info=drift_result, regime=regime)
            self.paper.record_snapshot(dt, prices)

            result["reason"] = "no_drift"
            result["drift"] = drift_result
            return result

        # Step 3: Generate execution plan
        logger.info("[2/5] Generating execution plan...")
        portfolio_value = self.paper.nav(prices)

        plan = generate_execution_plan(
            current_weights=current_weights,
            target_weights=target_weights,
            prices=prices,
            portfolio_value=portfolio_value,
            execution_reason=drift_result["trigger"],
            ticker_vols=ticker_vols,
        )

        if not plan.trades:
            logger.info("  No material trades — skipping")
            self.state_machine.reject_trade("no_material_trades")

            self.journal.log_no_trade(
                regime=regime,
                trigger="none",
                rationale="No material trades after plan generation",
            )

            self.paper.record_snapshot(dt, prices)
            result["reason"] = "no_trades"
            result["decision"] = "no_trade"
            return result

        # Step 4: Filter unnecessary trades
        trade_check = detect_unnecessary_trades(plan.trades)
        if trade_check["n_unnecessary"] > 0:
            logger.info(
                f"  Filtered {trade_check['n_unnecessary']} unnecessary trades"
            )
            plan.trades = trade_check["necessary"]

        # Step 5: Evaluate utility
        logger.info("[3/5] Evaluating utility...")
        utility = evaluate_rebalance_utility(
            current_weights=current_weights,
            target_weights=target_weights,
            trades=plan.trades,
            prices=prices,
            portfolio_value=portfolio_value,
            alpha_scores=alpha_scores,
            current_vol=current_vol,
            target_vol=target_vol,
            current_dr=current_dr,
            target_dr=target_dr,
            regime=regime,
            regime_changed=regime_changed,
            confidence=confidence,
        )

        result["utility"] = {
            "expected_gain": utility.expected_utility_gain,
            "friction": utility.total_friction,
            "net_utility": utility.net_utility,
            "should_trade": utility.should_trade,
        }

        if not utility.should_trade:
            logger.info(f"  Utility gate: SKIP — {utility.rationale}")
            self.state_machine.reject_trade("utility_negative")

            self.journal.log_no_trade(
                expected_utility=utility.expected_utility_gain,
                cost_estimate=utility.total_friction,
                confidence=confidence,
                regime=regime,
                trigger=drift_result["trigger"],
                rationale=utility.rationale,
            )

            self.paper.log_decision(
                dt, "no_trade", utility_estimate=utility,
                drift_info=drift_result, regime=regime,
            )
            self.paper.record_snapshot(dt, prices)

            result["decision"] = "no_trade"
            result["reason"] = utility.rationale
            return result

        # Step 6: Check turnover budget
        logger.info("[4/5] Checking turnover budget...")
        proposed_turnover = compute_turnover(current_weights, target_weights)
        month_key = f"{dt.year}-{dt.month:02d}"
        budget_check = self.turnover_budget.can_trade(proposed_turnover, month_key)

        if not budget_check["allowed"]:
            logger.info(
                f"  Turnover budget exceeded: "
                f"proposed={proposed_turnover:.3f}, "
                f"blocked_by={budget_check['blocked_by']}"
            )
            self.state_machine.reject_trade("turnover_budget")

            self.journal.log_no_trade(
                expected_utility=utility.expected_utility_gain,
                cost_estimate=utility.total_friction,
                confidence=confidence,
                regime=regime,
                trigger="turnover_budget",
                rationale=f"Turnover budget exceeded: {budget_check['blocked_by']}",
            )

            self.paper.record_snapshot(dt, prices)
            result["decision"] = "no_trade"
            result["reason"] = "turnover_budget"
            return result

        # Step 7: Execute
        logger.info("[5/5] Executing trades...")
        self.state_machine.approve_trade("utility_positive")

        exec_result = self.paper.execute_plan(plan, prices, country_map)
        self.turnover_budget.record_turnover(proposed_turnover, month_key)

        self.state_machine.execute_and_settle("trades_executed")

        # Log
        self.journal.log_trade(
            trades=exec_result.trades,
            expected_utility=utility.expected_utility_gain,
            cost_estimate=utility.total_friction,
            confidence=confidence,
            regime=regime,
            trigger=drift_result["trigger"],
            rationale=utility.rationale,
        )

        self.paper.log_decision(
            dt, "trade", utility_estimate=utility,
            drift_info=drift_result, regime=regime,
        )
        self.paper.record_snapshot(dt, prices, turnover=proposed_turnover)

        result["decision"] = "trade"
        result["reason"] = utility.rationale
        result["execution"] = {
            "n_trades": len(exec_result.trades),
            "total_notional": exec_result.total_notional,
            "total_slippage": exec_result.total_slippage,
            "total_fees": exec_result.total_fees,
            "total_tax": exec_result.total_tax,
            "efficiency": exec_result.execution_efficiency,
        }
        result["turnover"] = proposed_turnover

        logger.info(
            f"  EXECUTED: {len(exec_result.trades)} trades, "
            f"notional={exec_result.total_notional:,.0f}, "
            f"efficiency={exec_result.execution_efficiency:.4f}"
        )

        return result

    # ── Reports ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Full engine summary."""
        return {
            "paper_trading": self.paper.performance_summary(),
            "journal": self.journal.summary(),
            "turnover": self.turnover_budget.summary(),
            "state_machine": self.state_machine.summary(),
        }

    def save_all(self):
        """Save all artifacts."""
        self.paper.save()
        self.journal.save()
        logger.info("Execution engine artifacts saved")
