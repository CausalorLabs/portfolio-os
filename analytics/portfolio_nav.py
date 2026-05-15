"""
Portfolio NAV engine — computes daily INR-denominated portfolio value,
per-asset contributions, and daily returns.
"""

import pandas as pd
from loguru import logger


def calculate_portfolio_nav(
    inr_prices: pd.DataFrame,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily portfolio NAV in INR.

    Parameters
    ----------
    inr_prices : pd.DataFrame
        Output of convert_prices_to_inr (date, ticker, inr_price, …).
    holdings : pd.DataFrame
        Holdings with ticker and quantity columns.

    Returns
    -------
    pd.DataFrame
        Columns: date, portfolio_nav, daily_return
    """
    qty_map = dict(zip(holdings["ticker"], holdings["quantity"]))

    # Filter prices to held tickers only
    held = inr_prices[inr_prices["ticker"].isin(qty_map)].copy()

    if held.empty:
        logger.warning("No price data found for any held ticker")
        return pd.DataFrame(columns=["date", "portfolio_nav", "daily_return"])

    # Pivot to wide format and forward-fill so that each day has a price for
    # every asset (avoids NAV jumps when only some markets are open).
    wide = held.pivot_table(index="date", columns="ticker", values="inr_price", aggfunc="first")
    wide = wide.sort_index().ffill()

    # Only keep dates where ALL assets have a price (after the first full date)
    wide = wide.dropna()

    # Compute position values
    for ticker in wide.columns:
        wide[ticker] = wide[ticker] * qty_map[ticker]

    nav = pd.DataFrame({
        "date": wide.index,
        "portfolio_nav": wide.sum(axis=1).values,
    }).reset_index(drop=True)

    nav["daily_return"] = nav["portfolio_nav"].pct_change()

    logger.info(
        f"Portfolio NAV: {len(nav)} days, "
        f"₹{nav['portfolio_nav'].iloc[0]:,.0f} → ₹{nav['portfolio_nav'].iloc[-1]:,.0f}"
    )

    return nav


def calculate_asset_contributions(
    inr_prices: pd.DataFrame,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute each asset's daily weight (contribution %) in the portfolio.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, position_value, contribution_pct
    """
    qty_map = dict(zip(holdings["ticker"], holdings["quantity"]))
    held = inr_prices[inr_prices["ticker"].isin(qty_map)].copy()

    if held.empty:
        return pd.DataFrame(columns=["date", "ticker", "position_value", "contribution_pct"])

    held["position_value"] = held["ticker"].map(qty_map) * held["inr_price"]

    # Daily total
    daily_total = held.groupby("date")["position_value"].sum().rename("daily_total")
    held = held.merge(daily_total, on="date", how="left")
    held["contribution_pct"] = held["position_value"] / held["daily_total"] * 100

    result = held[["date", "ticker", "position_value", "contribution_pct"]].copy()
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Log latest snapshot
    latest = result[result["date"] == result["date"].max()]
    logger.info("Latest asset weights:")
    for _, row in latest.iterrows():
        logger.info(f"  {row['ticker']:15s} {row['contribution_pct']:6.2f}%")

    return result
