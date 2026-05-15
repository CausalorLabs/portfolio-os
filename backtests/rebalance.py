"""
Backtest rebalance engine — calculates the trades needed to move
from current portfolio state to target weights.

Respects cash availability and minimum trade sizes.
"""

import pandas as pd
from loguru import logger


MIN_TRADE_NOTIONAL = 500.0  # ignore trades smaller than ₹500


def calculate_target_positions(
    target_weights: dict[str, float],
    portfolio_nav: float,
    prices: dict[str, float],
    cash_reserve_pct: float = 0.0,
) -> dict[str, float]:
    """
    Convert target weights into target share quantities.

    Parameters
    ----------
    target_weights : dict
        ticker → target weight (sum ≤ 1.0).
    portfolio_nav : float
        Total portfolio value in INR.
    prices : dict
        ticker → current price in INR.
    cash_reserve_pct : float
        Fraction of NAV to keep as cash.

    Returns
    -------
    dict
        ticker → target quantity (fractional shares allowed).
    """
    investable = portfolio_nav * (1.0 - cash_reserve_pct)
    positions = {}
    for ticker, w in target_weights.items():
        if ticker.startswith("_"):
            continue
        p = prices.get(ticker, 0.0)
        if p > 0:
            positions[ticker] = (investable * w) / p
        else:
            positions[ticker] = 0.0
    return positions


def generate_trade_orders(
    current_holdings: dict[str, float],
    target_positions: dict[str, float],
    prices: dict[str, float],
    country_map: dict[str, str],
    min_notional: float = MIN_TRADE_NOTIONAL,
) -> list[dict]:
    """
    Generate trade orders to move from current to target positions.

    Parameters
    ----------
    current_holdings : dict
        ticker → current quantity.
    target_positions : dict
        ticker → target quantity.
    prices : dict
        ticker → price in INR.
    country_map : dict
        ticker → country code.
    min_notional : float
        Skip trades below this value.

    Returns
    -------
    list[dict]
        Each with: ticker, action, quantity, country.
    """
    all_tickers = set(current_holdings) | set(target_positions)
    orders = []

    for ticker in all_tickers:
        current = current_holdings.get(ticker, 0.0)
        target = target_positions.get(ticker, 0.0)
        delta = target - current
        price = prices.get(ticker, 0.0)
        notional = abs(delta * price)

        if notional < min_notional:
            continue

        if delta > 0:
            orders.append({
                "ticker": ticker,
                "action": "BUY",
                "quantity": delta,
                "country": country_map.get(ticker, "US"),
            })
        elif delta < 0:
            orders.append({
                "ticker": ticker,
                "action": "SELL",
                "quantity": abs(delta),
                "country": country_map.get(ticker, "US"),
            })

    if orders:
        logger.info(f"Trade orders: {len(orders)} trades generated")

    return orders
