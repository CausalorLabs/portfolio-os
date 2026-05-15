"""
Transaction cost engine — models realistic execution friction.

Conservative by default: assume higher costs, worse fills.
"""

import pandas as pd
from loguru import logger


# ── Default cost schedules ───────────────────────────────────────────────────

# Indian equities (NSE)
IN_EQUITY_COSTS = {
    "brokerage_pct": 0.0003,       # 0.03% (discount broker)
    "stt_pct": 0.001,              # 0.1% on sell delivery
    "exchange_fee_pct": 0.0000345, # NSE transaction charge
    "gst_on_brokerage": 0.18,      # 18% GST on brokerage
    "stamp_duty_pct": 0.00015,     # 0.015% on buy
}

# US equities
US_EQUITY_COSTS = {
    "brokerage_flat": 0.0,   # zero commission (most brokers)
    "fx_spread_pct": 0.005,  # 0.5% FX conversion spread
    "sec_fee_pct": 0.000008, # SEC fee on sells
}


def calculate_transaction_costs(
    ticker: str,
    quantity: float,
    price_inr: float,
    action: str,
    country: str,
) -> dict:
    """
    Calculate all-in transaction cost for a single trade.

    Parameters
    ----------
    ticker : str
    quantity : float
    price_inr : float
        Execution price in INR.
    action : str
        "BUY" or "SELL".
    country : str
        "IN" or "US".

    Returns
    -------
    dict
        Keys: brokerage, stt, exchange_fee, gst, stamp_duty,
              fx_cost, total_cost, cost_pct
    """
    notional = abs(quantity * price_inr)
    costs = {
        "brokerage": 0.0,
        "stt": 0.0,
        "exchange_fee": 0.0,
        "gst": 0.0,
        "stamp_duty": 0.0,
        "fx_cost": 0.0,
    }

    if country == "IN":
        costs["brokerage"] = notional * IN_EQUITY_COSTS["brokerage_pct"]
        costs["gst"] = costs["brokerage"] * IN_EQUITY_COSTS["gst_on_brokerage"]
        costs["exchange_fee"] = notional * IN_EQUITY_COSTS["exchange_fee_pct"]

        if action == "SELL":
            costs["stt"] = notional * IN_EQUITY_COSTS["stt_pct"]

        if action == "BUY":
            costs["stamp_duty"] = notional * IN_EQUITY_COSTS["stamp_duty_pct"]

    elif country == "US":
        costs["fx_cost"] = notional * US_EQUITY_COSTS["fx_spread_pct"]
        if action == "SELL":
            costs["brokerage"] = notional * US_EQUITY_COSTS["sec_fee_pct"]

    costs["total_cost"] = sum(costs.values())
    costs["cost_pct"] = costs["total_cost"] / notional if notional > 0 else 0.0

    return costs


def calculate_fx_conversion_cost(
    notional_inr: float,
    spread_pct: float = 0.005,
) -> float:
    """FX conversion cost for cross-border trades."""
    return abs(notional_inr) * spread_pct


def summarize_costs(trades: list[dict]) -> dict:
    """Aggregate cost breakdown across a set of trades."""
    keys = ["brokerage", "stt", "exchange_fee", "gst", "stamp_duty", "fx_cost", "total_cost"]
    summary = {k: sum(t.get(k, 0.0) for t in trades) for k in keys}
    total_notional = sum(abs(t.get("notional", 0.0)) for t in trades)
    summary["total_notional"] = total_notional
    summary["avg_cost_pct"] = (
        summary["total_cost"] / total_notional if total_notional > 0 else 0.0
    )

    logger.info(
        f"Transaction costs: ₹{summary['total_cost']:,.0f} "
        f"({summary['avg_cost_pct']:.3%} of ₹{total_notional:,.0f})"
    )
    return summary
