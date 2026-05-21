"""
Execution Engine — Utility-Based Rebalancing.

The most important question in portfolio management:

    "Should we trade at all?"

Trade ONLY if:
    Expected Utility Gain > Tax Cost + Slippage + Fees

This is the single biggest differentiator vs naive rebalancing.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class UtilityEstimate:
    """Result of utility-cost analysis for a proposed rebalance."""
    alpha_improvement: float = 0.0
    risk_reduction: float = 0.0
    diversification_gain: float = 0.0
    regime_urgency: float = 0.0
    expected_utility_gain: float = 0.0
    estimated_tax_cost: float = 0.0
    estimated_slippage: float = 0.0
    estimated_fees: float = 0.0
    total_friction: float = 0.0
    net_utility: float = 0.0
    should_trade: bool = False
    confidence: float = 0.0
    rationale: str = ""


# ── Utility Components ──────────────────────────────────────────────────────


def estimate_alpha_improvement(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    alpha_scores: dict[str, float] | None = None,
) -> float:
    """
    How much alpha do we gain by moving to target weights?

    Measures weighted alignment with alpha signal.
    """
    if not alpha_scores:
        return 0.0

    all_tickers = set(current_weights) | set(target_weights)
    current_alpha = sum(
        current_weights.get(t, 0) * alpha_scores.get(t, 0.5)
        for t in all_tickers
    )
    target_alpha = sum(
        target_weights.get(t, 0) * alpha_scores.get(t, 0.5)
        for t in all_tickers
    )

    return max(0, target_alpha - current_alpha)


def estimate_risk_reduction(
    current_vol: float,
    target_vol: float,
) -> float:
    """
    How much does portfolio risk decrease?

    Positive = risk decreases (good).
    """
    if current_vol <= 0:
        return 0.0
    return max(0, (current_vol - target_vol) / current_vol)


def estimate_diversification_gain(
    current_dr: float,
    target_dr: float,
) -> float:
    """Improvement in diversification ratio."""
    if current_dr <= 0:
        return 0.0
    return max(0, (target_dr - current_dr) / current_dr)


def estimate_regime_urgency(
    regime: str,
    regime_changed: bool,
) -> float:
    """
    How urgently does a regime shift require rebalancing?

    Panic regime + regime change = high urgency.
    """
    if not regime_changed:
        return 0.0

    urgency_map = {
        "panic": 0.8,
        "risk_off": 0.5,
        "high_vol": 0.4,
        "risk_on": 0.2,
    }
    return urgency_map.get(regime, 0.2)


# ── Friction Estimation ─────────────────────────────────────────────────────


def estimate_trade_friction(
    trades: list[dict],
    prices: dict[str, float],
    portfolio_value: float,
) -> dict:
    """
    Estimate total friction cost of a set of trades.

    Returns: {tax_cost, slippage, fees, total_friction, friction_pct}
    """
    cfg = _load_config()
    costs_cfg = cfg.get("costs", {})
    slippage_cfg = cfg.get("slippage", {})

    brokerage_pct = costs_cfg.get("brokerage_pct", 0.03) / 100
    stt_pct = costs_cfg.get("stt_pct", 0.10) / 100
    slippage_bps = slippage_cfg.get("base_bps", 10)
    penalty_bps = cfg.get("turnover", {}).get("trade_penalty_bps", 5)

    total_slippage = 0.0
    total_fees = 0.0
    total_tax_estimate = 0.0
    total_notional = 0.0

    for trade in trades:
        ticker = trade.get("ticker", "")
        qty = abs(trade.get("quantity", 0))
        price = prices.get(ticker, trade.get("price", 0))
        notional = qty * price
        total_notional += notional
        action = trade.get("action", "BUY")

        # Slippage
        slip = notional * slippage_bps / 10_000
        total_slippage += slip

        # Fees
        fee = notional * brokerage_pct
        if action == "SELL":
            fee += notional * stt_pct
        total_fees += fee

        # Tax estimate (rough — sells of profitable positions)
        if action == "SELL":
            cost_basis = trade.get("cost_basis", price)
            gain = (price - cost_basis) * qty
            if gain > 0:
                total_tax_estimate += gain * 0.15  # rough 15% estimate

    # Trade penalty
    penalty = total_notional * penalty_bps / 10_000

    total_friction = total_slippage + total_fees + total_tax_estimate + penalty
    friction_pct = total_friction / portfolio_value if portfolio_value > 0 else 0

    return {
        "tax_cost": total_tax_estimate,
        "slippage": total_slippage,
        "fees": total_fees,
        "trade_penalty": penalty,
        "total_friction": total_friction,
        "friction_pct": friction_pct,
        "total_notional": total_notional,
    }


# ── Core Utility Decision ───────────────────────────────────────────────────


def evaluate_rebalance_utility(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    trades: list[dict],
    prices: dict[str, float],
    portfolio_value: float,
    alpha_scores: dict[str, float] | None = None,
    current_vol: float = 0.0,
    target_vol: float = 0.0,
    current_dr: float = 1.0,
    target_dr: float = 1.0,
    regime: str = "risk_on",
    regime_changed: bool = False,
    confidence: float = 0.5,
) -> UtilityEstimate:
    """
    Master utility evaluation.

    Decides: should we trade at all?

    Trade ONLY if net_utility > 0, i.e.:
        expected_gain > total_friction
    """
    cfg = _load_config().get("utility_rebalancing", {})

    min_gain = cfg.get("min_utility_gain", 0.002)
    w_alpha = cfg.get("alpha_weight", 0.40)
    w_risk = cfg.get("risk_weight", 0.35)
    w_div = cfg.get("diversification_weight", 0.15)
    w_regime = cfg.get("regime_weight", 0.10)
    conf_floor = cfg.get("confidence_floor", 0.30)

    # Utility components
    alpha_imp = estimate_alpha_improvement(current_weights, target_weights, alpha_scores)
    risk_red = estimate_risk_reduction(current_vol, target_vol)
    div_gain = estimate_diversification_gain(current_dr, target_dr)
    regime_urg = estimate_regime_urgency(regime, regime_changed)

    expected_gain = (
        w_alpha * alpha_imp
        + w_risk * risk_red
        + w_div * div_gain
        + w_regime * regime_urg
    )

    # Friction
    friction = estimate_trade_friction(trades, prices, portfolio_value)

    net = expected_gain - friction["friction_pct"]

    # Decision
    should_trade = (
        net > min_gain
        and confidence >= conf_floor
    )

    # Rationale
    if not should_trade:
        if confidence < conf_floor:
            rationale = f"Confidence too low ({confidence:.2f} < {conf_floor:.2f})"
        elif net <= 0:
            rationale = f"Friction ({friction['friction_pct']:.4f}) exceeds utility gain ({expected_gain:.4f})"
        else:
            rationale = f"Net utility ({net:.4f}) below minimum ({min_gain:.4f})"
    else:
        rationale = (
            f"Net utility {net:.4f}: "
            f"alpha={alpha_imp:.3f}, risk={risk_red:.3f}, "
            f"div={div_gain:.3f}, regime={regime_urg:.3f}, "
            f"friction={friction['friction_pct']:.4f}"
        )

    return UtilityEstimate(
        alpha_improvement=alpha_imp,
        risk_reduction=risk_red,
        diversification_gain=div_gain,
        regime_urgency=regime_urg,
        expected_utility_gain=expected_gain,
        estimated_tax_cost=friction["tax_cost"],
        estimated_slippage=friction["slippage"],
        estimated_fees=friction["fees"],
        total_friction=friction["total_friction"],
        net_utility=net,
        should_trade=should_trade,
        confidence=confidence,
        rationale=rationale,
    )
