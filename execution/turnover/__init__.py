"""
Execution Engine — Turnover Control.

Most quant systems die from overtrading, not from bad signals.

Controls:
  - Monthly/annual turnover budgets
  - Trade penalty functions
  - Signal stability requirements
  - Unnecessary trade detection
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/execution_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ── Turnover Measurement ───────────────────────────────────────────────────


def compute_turnover(
    prev_weights: dict[str, float],
    new_weights: dict[str, float],
) -> float:
    """One-way turnover between two weight vectors."""
    all_tickers = set(prev_weights) | set(new_weights)
    total = sum(
        abs(prev_weights.get(t, 0) - new_weights.get(t, 0))
        for t in all_tickers if not t.startswith("_")
    )
    return total / 2


# ── Turnover Budget ────────────────────────────────────────────────────────


class TurnoverBudget:
    """
    Enforces monthly and annual turnover limits.

    Prevents signal noise from causing overtrading.
    """

    def __init__(self):
        cfg = _load_config().get("turnover", {})
        self.monthly_budget = cfg.get("monthly_budget", 0.20)
        self.annual_budget = cfg.get("annual_budget", 1.50)
        self._monthly_used: dict[str, float] = {}  # YYYY-MM → turnover
        self._annual_used: float = 0.0

    def remaining_monthly(self, month_key: str) -> float:
        """How much turnover budget remains this month."""
        used = self._monthly_used.get(month_key, 0)
        return max(0, self.monthly_budget - used)

    def remaining_annual(self) -> float:
        """How much turnover budget remains this year."""
        return max(0, self.annual_budget - self._annual_used)

    def can_trade(self, proposed_turnover: float, month_key: str) -> dict:
        """Check if proposed turnover fits within budget."""
        monthly_ok = proposed_turnover <= self.remaining_monthly(month_key)
        annual_ok = proposed_turnover <= self.remaining_annual()

        return {
            "allowed": monthly_ok and annual_ok,
            "proposed": proposed_turnover,
            "monthly_remaining": self.remaining_monthly(month_key),
            "annual_remaining": self.remaining_annual(),
            "blocked_by": (
                "none" if monthly_ok and annual_ok
                else "monthly_budget" if not monthly_ok
                else "annual_budget"
            ),
        }

    def record_turnover(self, turnover: float, month_key: str):
        """Record executed turnover against budgets."""
        self._monthly_used[month_key] = self._monthly_used.get(month_key, 0) + turnover
        self._annual_used += turnover

    def summary(self) -> dict:
        return {
            "monthly_budget": self.monthly_budget,
            "annual_budget": self.annual_budget,
            "monthly_used": dict(self._monthly_used),
            "annual_used": self._annual_used,
        }


# ── Signal Stability ───────────────────────────────────────────────────────


def check_signal_stability(
    signal_history: pd.DataFrame,
    ticker: str,
    window: int | None = None,
) -> dict:
    """
    Check if a signal has been stable enough to warrant trading.

    Don't trade on noisy, unstable signals.

    Parameters
    ----------
    signal_history : DataFrame with date, ticker, signal columns
    ticker : ticker to check
    window : stability window in days

    Returns
    -------
    {is_stable, avg_signal, signal_vol, n_flips}
    """
    cfg = _load_config().get("turnover", {})
    if window is None:
        window = cfg.get("signal_stability_window", 5)

    ticker_data = signal_history[signal_history["ticker"] == ticker].tail(window)

    if len(ticker_data) < window:
        return {"is_stable": False, "reason": "insufficient_data"}

    signals = ticker_data["signal"] if "signal" in ticker_data.columns else ticker_data.iloc[:, -1]

    avg = float(signals.mean())
    vol = float(signals.std())

    # Count direction flips
    diffs = signals.diff().dropna()
    flips = int((diffs.abs() > 0.1).sum())

    # Stable = low volatility and few flips
    is_stable = vol < 0.15 and flips <= window // 2

    return {
        "is_stable": is_stable,
        "avg_signal": avg,
        "signal_vol": vol,
        "n_flips": flips,
        "window": window,
    }


# ── Trade Penalty ──────────────────────────────────────────────────────────


def compute_trade_penalty(
    trades: list[dict],
    portfolio_value: float,
) -> dict:
    """
    Compute penalty for proposed trades.

    Penalizes:
      - Small unnecessary changes (below min_notional)
      - High number of trades (complexity cost)
    """
    cfg = _load_config().get("turnover", {})
    penalty_bps = cfg.get("trade_penalty_bps", 5)
    min_notional = cfg.get("min_trade_notional", 500)

    total_notional = sum(t.get("notional", 0) for t in trades)
    n_trades = len(trades)
    n_small = sum(1 for t in trades if t.get("notional", 0) < min_notional * 2)

    # Base penalty
    base = total_notional * penalty_bps / 10_000

    # Small trade surcharge (discourage micro-optimization)
    small_surcharge = n_small * 100  # ₹100 per tiny trade

    # Complexity penalty
    complexity = max(0, n_trades - 4) * 50  # ₹50 per trade beyond 4

    total_penalty = base + small_surcharge + complexity
    penalty_pct = total_penalty / max(portfolio_value, 1)

    return {
        "base_penalty": base,
        "small_trade_surcharge": small_surcharge,
        "complexity_penalty": complexity,
        "total_penalty": total_penalty,
        "penalty_pct": penalty_pct,
        "n_small_trades": n_small,
        "n_trades": n_trades,
    }


# ── Unnecessary Trade Detection ────────────────────────────────────────────


def detect_unnecessary_trades(
    trades: list[dict],
    min_weight_change: float = 0.01,
) -> dict:
    """
    Flag trades that are likely unnecessary (micro-optimization).

    A trade is unnecessary if:
      - Weight change < 1%
      - Notional < 2x minimum
    """
    cfg = _load_config().get("turnover", {})
    min_notional = cfg.get("min_trade_notional", 500)

    necessary = []
    unnecessary = []

    for trade in trades:
        weight_change = abs(trade.get("weight_change", 0))
        notional = trade.get("notional", 0)

        if weight_change < min_weight_change or notional < min_notional * 2:
            unnecessary.append(trade)
        else:
            necessary.append(trade)

    return {
        "necessary": necessary,
        "unnecessary": unnecessary,
        "n_necessary": len(necessary),
        "n_unnecessary": len(unnecessary),
        "pct_unnecessary": len(unnecessary) / max(len(trades), 1),
    }
