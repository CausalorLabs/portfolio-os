"""
Execution simulator — models realistic fills with slippage and spread.

Conservative by default: assume worse fills, not idealized ones.
"""

import pandas as pd
from loguru import logger


DEFAULT_SLIPPAGE_BPS = 10  # 10 basis points = 0.10%


def apply_slippage(
    price: float,
    action: str,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> float:
    """
    Apply slippage to a market price.

    BUY  → price goes UP   (you pay more)
    SELL → price goes DOWN  (you receive less)
    """
    slip = slippage_bps / 10_000
    if action == "BUY":
        return price * (1.0 + slip)
    else:
        return price * (1.0 - slip)


def simulate_execution(
    orders: list[dict],
    prices: dict[str, float],
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> list[dict]:
    """
    Simulate execution of a list of trade orders.

    Parameters
    ----------
    orders : list[dict]
        Each dict has: ticker, action, quantity, country.
    prices : dict[str, float]
        ticker → market price (INR).
    slippage_bps : float
        Slippage in basis points.

    Returns
    -------
    list[dict]
        Executed trades with market_price, execution_price, slippage_cost.
    """
    executed = []

    for order in orders:
        ticker = order["ticker"]
        action = order["action"]
        qty = order["quantity"]
        market_price = prices.get(ticker, 0.0)

        if market_price <= 0 or qty <= 0:
            continue

        exec_price = apply_slippage(market_price, action, slippage_bps)
        slippage_cost = abs(exec_price - market_price) * qty

        executed.append({
            "ticker": ticker,
            "action": action,
            "quantity": qty,
            "market_price": market_price,
            "execution_price": exec_price,
            "slippage_cost": slippage_cost,
            "notional": exec_price * qty,
            "country": order.get("country", "US"),
        })

    total_slippage = sum(t["slippage_cost"] for t in executed)
    if executed:
        logger.info(
            f"Executed {len(executed)} trades, "
            f"slippage: ₹{total_slippage:,.0f} ({slippage_bps} bps)"
        )

    return executed
