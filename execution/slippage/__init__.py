"""
Execution Engine — Slippage & Market Impact.

Estimates true execution cost, not just theoretical price.

Models:
  - Simple: fixed bps
  - Volatility-adjusted: higher vol → higher slippage
  - Large order impact: big orders move the market
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


def estimate_slippage_simple(
    price: float,
    action: str,
    slippage_bps: int | None = None,
) -> dict:
    """
    Fixed bps slippage model.

    BUY: pay more. SELL: receive less.
    """
    cfg = _load_config().get("slippage", {})
    if slippage_bps is None:
        slippage_bps = cfg.get("base_bps", 10)

    slip_pct = slippage_bps / 10_000
    if action == "BUY":
        exec_price = price * (1 + slip_pct)
    else:
        exec_price = price * (1 - slip_pct)

    return {
        "market_price": price,
        "execution_price": exec_price,
        "slippage_bps": slippage_bps,
        "slippage_cost": abs(exec_price - price),
    }


def estimate_slippage_vol_adjusted(
    price: float,
    action: str,
    ticker_vol: float,
    avg_vol: float = 0.15,
) -> dict:
    """
    Volatility-adjusted slippage.

    slippage = base_bps × (1 + vol_mult × ticker_vol / avg_vol)

    High-vol assets cost more to trade.
    """
    cfg = _load_config().get("slippage", {})
    base_bps = cfg.get("base_bps", 10)
    vol_mult = cfg.get("vol_multiplier", 2.0)

    if avg_vol <= 0:
        avg_vol = 0.15

    vol_ratio = ticker_vol / avg_vol
    adjusted_bps = base_bps * (1 + vol_mult * max(0, vol_ratio - 1))
    adjusted_bps = min(adjusted_bps, 100)  # cap at 100bps

    slip_pct = adjusted_bps / 10_000
    if action == "BUY":
        exec_price = price * (1 + slip_pct)
    else:
        exec_price = price * (1 - slip_pct)

    return {
        "market_price": price,
        "execution_price": exec_price,
        "slippage_bps": adjusted_bps,
        "slippage_cost": abs(exec_price - price),
        "vol_ratio": vol_ratio,
    }


def estimate_market_impact(
    trade_value: float,
    avg_daily_volume_value: float,
) -> dict:
    """
    Market impact estimate for large orders.

    If trade > threshold % of ADV, add extra slippage.
    """
    cfg = _load_config().get("slippage", {})
    threshold_pct = cfg.get("large_order_threshold_pct", 5.0) / 100
    impact_bps = cfg.get("market_impact_bps", 5)

    if avg_daily_volume_value <= 0:
        return {"is_large": False, "impact_bps": 0, "impact_cost": 0}

    pct_of_adv = trade_value / avg_daily_volume_value

    if pct_of_adv > threshold_pct:
        # Square-root market impact model
        impact = impact_bps * np.sqrt(pct_of_adv / threshold_pct)
        impact_cost = trade_value * impact / 10_000
        return {
            "is_large": True,
            "pct_of_adv": pct_of_adv,
            "impact_bps": impact,
            "impact_cost": impact_cost,
        }

    return {"is_large": False, "pct_of_adv": pct_of_adv, "impact_bps": 0, "impact_cost": 0}


def estimate_execution_cost(
    ticker: str,
    price: float,
    quantity: float,
    action: str,
    ticker_vol: float = 0.15,
    avg_daily_volume_value: float = 0,
) -> dict:
    """
    Full execution cost estimate for a single trade.

    Returns: {
        ticker, action, quantity, market_price, execution_price,
        slippage_bps, slippage_cost, market_impact, total_cost,
        liquidity_score
    }
    """
    cfg = _load_config().get("slippage", {})
    model = cfg.get("model", "volatility_adjusted")

    if model == "volatility_adjusted":
        slip = estimate_slippage_vol_adjusted(price, action, ticker_vol)
    else:
        slip = estimate_slippage_simple(price, action)

    trade_value = quantity * price
    impact = estimate_market_impact(trade_value, avg_daily_volume_value)

    total_slip = slip["slippage_cost"] * quantity
    total_impact = impact.get("impact_cost", 0)

    # Liquidity score: 0-1 (1 = very liquid)
    if avg_daily_volume_value > 0:
        liquidity = min(1.0, avg_daily_volume_value / (trade_value * 20))
    else:
        liquidity = 0.5  # unknown

    return {
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "market_price": price,
        "execution_price": slip["execution_price"],
        "slippage_bps": slip["slippage_bps"],
        "slippage_cost": total_slip,
        "market_impact": total_impact,
        "total_execution_cost": total_slip + total_impact,
        "liquidity_score": liquidity,
    }
