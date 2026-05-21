"""
Execution Engine — Execution Simulator.

Generates execution plans and simulates trade outcomes.

Output:
  - execution_plan: ticker | side | quantity | estimated_price |
                    estimated_cost | priority | execution_reason
  - execution_result: expected vs actual fill, realized slippage,
                      execution efficiency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from omegaconf import OmegaConf

from execution.slippage import estimate_execution_cost


def _load_config() -> dict:
    cfg_path = Path("configs/execution_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class ExecutionPlan:
    """A planned set of trades with priority and cost estimates."""
    date: date
    trades: list[dict] = field(default_factory=list)
    total_notional: float = 0.0
    total_estimated_cost: float = 0.0
    n_buys: int = 0
    n_sells: int = 0


@dataclass
class ExecutionResult:
    """Outcome of simulated execution."""
    date: date
    trades: list[dict] = field(default_factory=list)
    total_notional: float = 0.0
    total_slippage: float = 0.0
    total_fees: float = 0.0
    total_tax: float = 0.0
    execution_efficiency: float = 1.0


# ── Execution Plan Generation ───────────────────────────────────────────────


def generate_execution_plan(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    prices: dict[str, float],
    portfolio_value: float,
    execution_reason: str = "rebalance",
    ticker_vols: dict[str, float] | None = None,
) -> ExecutionPlan:
    """
    Generate prioritized execution plan from weight differences.

    Priority rules:
      1. Sells before buys (raise cash first)
      2. Largest weight changes first
      3. Filter out sub-minimum trades
    """
    cfg = _load_config()
    min_notional = cfg.get("turnover", {}).get("min_trade_notional", 500)

    trades = []
    all_tickers = set(current_weights) | set(target_weights)

    for ticker in all_tickers:
        if ticker.startswith("_"):
            continue

        curr = current_weights.get(ticker, 0)
        tgt = target_weights.get(ticker, 0)
        delta = tgt - curr

        if abs(delta) < 0.001:
            continue

        price = prices.get(ticker, 0)
        if price <= 0:
            continue

        trade_value = abs(delta) * portfolio_value
        if trade_value < min_notional:
            continue

        quantity = trade_value / price
        action = "BUY" if delta > 0 else "SELL"

        vol = (ticker_vols or {}).get(ticker, 0.15)
        cost_est = estimate_execution_cost(
            ticker, price, quantity, action, ticker_vol=vol,
        )

        trades.append({
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": price,
            "notional": trade_value,
            "weight_change": delta,
            "estimated_slippage": cost_est["slippage_cost"],
            "estimated_cost": cost_est["total_execution_cost"],
            "liquidity_score": cost_est["liquidity_score"],
            "execution_reason": execution_reason,
            # Priority: sells first (0), then by absolute weight change
            "priority": (0 if action == "SELL" else 1, -abs(delta)),
        })

    trades.sort(key=lambda t: t["priority"])

    # Assign sequential priority numbers
    for i, trade in enumerate(trades):
        trade["priority"] = i + 1

    plan = ExecutionPlan(
        date=date.today(),
        trades=trades,
        total_notional=sum(t["notional"] for t in trades),
        total_estimated_cost=sum(t["estimated_cost"] for t in trades),
        n_buys=sum(1 for t in trades if t["action"] == "BUY"),
        n_sells=sum(1 for t in trades if t["action"] == "SELL"),
    )

    logger.info(
        f"  Execution plan: {len(trades)} trades, "
        f"notional={plan.total_notional:,.0f}, "
        f"cost={plan.total_estimated_cost:,.0f}"
    )

    return plan


# ── Simulated Execution ────────────────────────────────────────────────────


def simulate_execution(
    plan: ExecutionPlan,
    portfolio_state,
    prices: dict[str, float],
    country_map: dict[str, str] | None = None,
) -> ExecutionResult:
    """
    Simulate executing a plan against portfolio state.

    Applies slippage, computes costs/taxes, updates state.
    """
    from backtests.costs import calculate_transaction_costs
    from backtests.taxes import calculate_capital_gains_tax

    if country_map is None:
        country_map = {}

    executed = []
    total_slippage = 0.0
    total_fees = 0.0
    total_tax = 0.0
    total_notional = 0.0

    for trade in plan.trades:
        ticker = trade["ticker"]
        action = trade["action"]
        quantity = trade["quantity"]
        market_price = prices.get(ticker, trade["price"])
        country = country_map.get(ticker, "IN")

        # Apply slippage
        slip = trade.get("estimated_slippage", 0)
        if action == "BUY":
            exec_price = market_price * (1 + slip / max(market_price * quantity, 1))
        else:
            exec_price = market_price * (1 - slip / max(market_price * quantity, 1))

        slippage_cost = abs(exec_price - market_price) * quantity
        total_slippage += slippage_cost

        # Transaction costs
        costs = calculate_transaction_costs(
            ticker, quantity, exec_price, action, country,
        )
        fee = costs.get("total_cost", 0)
        total_fees += fee

        # Tax on sells
        tax = 0.0
        realized_pnl = 0.0
        if action == "SELL":
            cost_basis = portfolio_state.cost_basis(ticker)
            if cost_basis > 0:
                gain = (exec_price - cost_basis) * quantity
                if gain > 0:
                    tax_info = calculate_capital_gains_tax(
                        exec_price, cost_basis, quantity,
                        pd.Timestamp.now() - pd.Timedelta(days=365),
                        pd.Timestamp.now(), country,
                    )
                    tax = tax_info.get("tax", 0)
                    total_tax += tax

            realized_pnl = portfolio_state.sell(
                ticker, quantity, exec_price,
                pd.Timestamp(plan.date), cost=fee, tax=tax,
            )
        else:
            portfolio_state.buy(
                ticker, quantity, exec_price,
                pd.Timestamp(plan.date), cost=fee,
            )

        notional = quantity * exec_price
        total_notional += notional

        executed.append({
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "market_price": market_price,
            "execution_price": exec_price,
            "notional": notional,
            "slippage_cost": slippage_cost,
            "transaction_fee": fee,
            "tax": tax,
            "realized_pnl": realized_pnl,
            "total_friction": slippage_cost + fee + tax,
        })

    total_friction = total_slippage + total_fees + total_tax
    efficiency = 1 - (total_friction / max(total_notional, 1))

    result = ExecutionResult(
        date=plan.date,
        trades=executed,
        total_notional=total_notional,
        total_slippage=total_slippage,
        total_fees=total_fees,
        total_tax=total_tax,
        execution_efficiency=efficiency,
    )

    logger.info(
        f"  Execution result: {len(executed)} trades, "
        f"friction={total_friction:,.0f}, "
        f"efficiency={efficiency:.4f}"
    )

    return result
