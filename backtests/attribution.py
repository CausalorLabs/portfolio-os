"""
Attribution engine — decomposes backtest performance into
explainable components.

Gross Return → Costs → Taxes → FX Drag → Net Return
"""

import pandas as pd
import numpy as np
from loguru import logger


def calculate_performance_attribution(
    nav_series: pd.DataFrame,
    ledger_df: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """
    Decompose backtest performance into gross and friction components.

    Parameters
    ----------
    nav_series : pd.DataFrame
        Must have columns: date, nav, total_costs, total_taxes.
    ledger_df : pd.DataFrame
        Trade ledger from TradeLedger.to_dataframe().
    initial_capital : float

    Returns
    -------
    dict
        Attribution breakdown.
    """
    if nav_series.empty:
        return {}

    final_nav = nav_series["nav"].iloc[-1]
    n_days = len(nav_series)
    years = n_days / 252

    # Gross vs net
    total_costs = nav_series["total_costs"].iloc[-1] if "total_costs" in nav_series else 0.0
    total_taxes = nav_series["total_taxes"].iloc[-1] if "total_taxes" in nav_series else 0.0

    gross_nav = final_nav + total_costs + total_taxes
    gross_return = (gross_nav / initial_capital) - 1
    net_return = (final_nav / initial_capital) - 1

    gross_cagr = (gross_nav / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    net_cagr = (final_nav / initial_capital) ** (1 / years) - 1 if years > 0 else 0

    # Friction breakdown from ledger
    slippage_drag = 0.0
    cost_drag = 0.0
    tax_drag = 0.0
    if not ledger_df.empty:
        slippage_drag = ledger_df["slippage_cost"].sum()
        cost_drag = ledger_df["transaction_cost"].sum()
        tax_drag = ledger_df["tax"].sum()

    total_friction = slippage_drag + cost_drag + tax_drag

    attribution = {
        "initial_capital": initial_capital,
        "final_nav": final_nav,
        "gross_nav": gross_nav,
        "gross_return": gross_return,
        "net_return": net_return,
        "gross_cagr": gross_cagr,
        "net_cagr": net_cagr,
        "slippage_drag": slippage_drag,
        "cost_drag": cost_drag,
        "tax_drag": tax_drag,
        "total_friction": total_friction,
        "friction_as_pct_of_capital": total_friction / initial_capital,
        "friction_cagr_drag": gross_cagr - net_cagr,
        "n_days": n_days,
        "years": years,
    }

    _log_attribution(attribution)
    return attribution


def calculate_allocation_attribution(
    nav_series: pd.DataFrame,
    rebalance_log: list[dict],
) -> pd.DataFrame:
    """
    Attribute returns to allocation decisions at each rebalance.

    Returns
    -------
    pd.DataFrame
        Per-rebalance period attribution.
    """
    if not rebalance_log or nav_series.empty:
        return pd.DataFrame()

    periods = []
    for i, rebal in enumerate(rebalance_log):
        date = rebal["date"]
        nav_at = nav_series[nav_series["date"] == date]
        if nav_at.empty:
            # Find nearest
            nav_at = nav_series[nav_series["date"] <= date]
        if nav_at.empty:
            continue

        nav_val = nav_at.iloc[-1]["nav"]

        periods.append({
            "rebalance_date": date,
            "n_trades": rebal.get("n_trades", 0),
            "turnover": rebal.get("turnover", 0.0),
            "nav": nav_val,
        })

    if not periods:
        return pd.DataFrame()

    df = pd.DataFrame(periods)
    df["period_return"] = df["nav"].pct_change()
    return df


def _log_attribution(attr: dict) -> None:
    """Log a readable attribution summary."""
    logger.info("Performance Attribution:")
    logger.info(f"  Gross CAGR:       {attr['gross_cagr']:+.2%}")
    logger.info(f"  Net CAGR:         {attr['net_cagr']:+.2%}")
    logger.info(f"  Friction drag:    {attr['friction_cagr_drag']:.2%}")
    logger.info(f"  ├─ Slippage:      ₹{attr['slippage_drag']:>10,.0f}")
    logger.info(f"  ├─ Costs:         ₹{attr['cost_drag']:>10,.0f}")
    logger.info(f"  └─ Taxes:         ₹{attr['tax_drag']:>10,.0f}")
    logger.info(f"  Total friction:   ₹{attr['total_friction']:>10,.0f} "
                f"({attr['friction_as_pct_of_capital']:.2%} of capital)")
