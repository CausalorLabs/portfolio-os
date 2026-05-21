"""
Execution Engine — Paper Trading.

Operates a shadow portfolio continuously:

  Ingest → Features → Regime → Alpha → Risk → Evaluate Rebalance →
  Simulate Execution → Update State → Log

This is the first version of a true portfolio operating engine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from omegaconf import OmegaConf

from backtests.portfolio_state import PortfolioState


def _load_config() -> dict:
    cfg_path = Path("configs/execution_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class PaperPortfolioSnapshot:
    """Daily snapshot of the paper portfolio."""
    date: date
    cash: float
    market_value: float
    nav: float
    n_positions: int
    turnover: float
    realized_pnl: float
    unrealized_pnl: float
    total_costs: float
    total_taxes: float
    tax_liability: float


class PaperTradingEngine:
    """
    Shadow portfolio engine for paper trading.

    Maintains a full PortfolioState without real execution.
    Tracks NAV, PnL, costs, taxes daily.
    """

    def __init__(self, initial_capital: float | None = None):
        cfg = _load_config().get("paper_trading", {})
        if initial_capital is None:
            initial_capital = cfg.get("initial_capital", 10_000_000)

        self.state = PortfolioState(cash=initial_capital)
        self.initial_capital = initial_capital
        self.history: list[PaperPortfolioSnapshot] = []
        self.trade_log: list[dict] = []
        self.decision_log: list[dict] = []
        self._last_weights: dict[str, float] = {}

        logger.info(f"Paper trading initialized: ₹{initial_capital:,.0f}")

    # ── State Queries ───────────────────────────────────────────────────

    def nav(self, prices: dict[str, float]) -> float:
        return self.state.nav(prices)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        return self.state.weights(prices)

    def unrealized_pnl(self, prices: dict[str, float]) -> dict[str, float]:
        return self.state.unrealized_pnl(prices)

    def positions(self) -> dict[str, float]:
        return dict(self.state.holdings)

    # ── Daily Update ────────────────────────────────────────────────────

    def record_snapshot(
        self,
        dt: date,
        prices: dict[str, float],
        turnover: float = 0.0,
    ) -> PaperPortfolioSnapshot:
        """Record daily portfolio state."""
        state_snap = self.state.snapshot(pd.Timestamp(dt), prices)

        snapshot = PaperPortfolioSnapshot(
            date=dt,
            cash=state_snap["cash"],
            market_value=state_snap["market_value"],
            nav=state_snap["nav"],
            n_positions=state_snap["n_positions"],
            turnover=turnover,
            realized_pnl=state_snap["realized_pnl"],
            unrealized_pnl=state_snap["unrealized_pnl"],
            total_costs=state_snap["total_costs"],
            total_taxes=state_snap["total_taxes"],
            tax_liability=self._estimate_tax_liability(prices),
        )

        self.history.append(snapshot)
        return snapshot

    def _estimate_tax_liability(self, prices: dict[str, float]) -> float:
        """Estimate unrealized tax liability if all positions sold today."""
        unrealized = self.state.unrealized_pnl(prices)
        gains = sum(v for v in unrealized.values() if v > 0)
        return gains * 0.15  # rough estimate

    # ── Execution ───────────────────────────────────────────────────────

    def execute_plan(
        self,
        plan,
        prices: dict[str, float],
        country_map: dict[str, str] | None = None,
    ):
        """Execute a plan against the paper portfolio."""
        from execution.simulation import simulate_execution

        result = simulate_execution(plan, self.state, prices, country_map)

        # Track turnover
        current_w = self.weights(prices)
        if self._last_weights:
            turnover = sum(
                abs(current_w.get(t, 0) - self._last_weights.get(t, 0))
                for t in set(current_w) | set(self._last_weights)
                if not t.startswith("_")
            ) / 2
        else:
            turnover = 0

        self._last_weights = current_w
        self.trade_log.extend(result.trades)

        return result

    # ── Decision Logging ────────────────────────────────────────────────

    def log_decision(
        self,
        dt: date,
        decision: str,
        utility_estimate=None,
        drift_info: dict | None = None,
        regime: str = "risk_on",
    ):
        """Log a rebalance decision (trade or no-trade)."""
        entry = {
            "date": dt,
            "decision": decision,
            "regime": regime,
        }

        if utility_estimate is not None:
            entry.update({
                "expected_utility": utility_estimate.expected_utility_gain,
                "total_friction": utility_estimate.total_friction,
                "net_utility": utility_estimate.net_utility,
                "rationale": utility_estimate.rationale,
            })

        if drift_info:
            entry["max_drift"] = drift_info.get("drift", {}).get("max_drift", 0)
            entry["trigger"] = drift_info.get("trigger", "none")

        self.decision_log.append(entry)

    # ── Reports ─────────────────────────────────────────────────────────

    def nav_history(self) -> pd.DataFrame:
        """NAV time series."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([
            {"date": s.date, "nav": s.nav} for s in self.history
        ])

    def performance_summary(self) -> dict:
        """Summary of paper trading performance."""
        if not self.history:
            return {"status": "no_data"}

        first = self.history[0]
        last = self.history[-1]
        days = (last.date - first.date).days

        total_return = (last.nav / self.initial_capital) - 1
        ann_return = (1 + total_return) ** (365 / max(days, 1)) - 1

        nav_series = [s.nav for s in self.history]
        peak = nav_series[0]
        max_dd = 0
        for v in nav_series:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)

        n_trades = len(self.trade_log)
        n_decisions = len(self.decision_log)
        n_no_trade = sum(1 for d in self.decision_log if d["decision"] == "no_trade")

        return {
            "start_date": first.date,
            "end_date": last.date,
            "days": days,
            "initial_capital": self.initial_capital,
            "final_nav": last.nav,
            "total_return": total_return,
            "annualized_return": ann_return,
            "max_drawdown": max_dd,
            "total_costs": last.total_costs,
            "total_taxes": last.total_taxes,
            "total_friction": last.total_costs + last.total_taxes,
            "n_trades": n_trades,
            "n_decisions": n_decisions,
            "n_no_trade": n_no_trade,
            "trade_skip_ratio": n_no_trade / max(n_decisions, 1),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Full history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": s.date,
                "cash": s.cash,
                "market_value": s.market_value,
                "nav": s.nav,
                "n_positions": s.n_positions,
                "turnover": s.turnover,
                "realized_pnl": s.realized_pnl,
                "unrealized_pnl": s.unrealized_pnl,
                "total_costs": s.total_costs,
                "total_taxes": s.total_taxes,
                "tax_liability": s.tax_liability,
            }
            for s in self.history
        ])

    def save(self, path: str = "data/processed/paper_portfolio.parquet"):
        """Save paper portfolio history to parquet."""
        df = self.to_dataframe()
        if not df.empty:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info(f"Paper portfolio saved: {path}")
