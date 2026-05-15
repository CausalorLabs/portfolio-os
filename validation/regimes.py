"""
Market regime analysis — evaluate strategy behavior across
bull/bear/volatile/sideways/FX-stress environments.

Regimes are identified from price data, then strategy performance
is decomposed by regime to detect fragile alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from analytics.metrics import (
    calculate_cagr,
    calculate_sharpe,
    calculate_volatility,
    calculate_max_drawdown,
)


def identify_market_regimes(
    wide_prices: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    vol_window: int = 60,
    trend_window: int = 120,
) -> pd.DataFrame:
    """
    Classify each trading day into a market regime.

    Regimes:
        - bull: positive trend + low vol
        - bear: negative trend + any vol
        - high_vol: high volatility (top quartile)
        - sideways: flat trend + low/moderate vol

    Parameters
    ----------
    wide_prices : pd.DataFrame
        Wide-format daily prices (columns = tickers, index = dates).
    benchmark_ticker : str
        Ticker to use as market proxy for regime detection.
    vol_window : int
        Rolling window for volatility estimation.
    trend_window : int
        Rolling window for trend detection.

    Returns
    -------
    pd.DataFrame
        Columns: date, regime, rolling_return, rolling_vol.
    """
    if benchmark_ticker not in wide_prices.columns:
        # Use equal-weight portfolio return as proxy
        returns = wide_prices.pct_change().dropna()
        market_ret = returns.mean(axis=1)
    else:
        market_ret = wide_prices[benchmark_ticker].pct_change().dropna()

    rolling_ret = market_ret.rolling(trend_window).mean() * 252
    rolling_vol = market_ret.rolling(vol_window).std() * np.sqrt(252)

    df = pd.DataFrame({
        "date": market_ret.index,
        "daily_return": market_ret.values,
    }).set_index("date")

    df["rolling_return"] = rolling_ret
    df["rolling_vol"] = rolling_vol
    df = df.dropna()

    # Regime classification
    vol_median = df["rolling_vol"].median()
    vol_q75 = df["rolling_vol"].quantile(0.75)

    conditions = [
        (df["rolling_return"] > 0.02) & (df["rolling_vol"] <= vol_median),
        (df["rolling_return"] < -0.02),
        (df["rolling_vol"] > vol_q75),
    ]
    choices = ["bull", "bear", "high_vol"]
    df["regime"] = np.select(conditions, choices, default="sideways")

    df = df.reset_index()

    regime_counts = df["regime"].value_counts()
    logger.info("Market Regimes Identified:")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {regime:12s}: {count:>5} days ({pct:.1f}%)")

    return df[["date", "regime", "rolling_return", "rolling_vol"]]


def evaluate_regime_performance(
    nav_series: pd.DataFrame,
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose portfolio performance by market regime.

    Parameters
    ----------
    nav_series : pd.DataFrame
        Portfolio NAV with columns: date, portfolio_nav (or nav), daily_return.
    regimes : pd.DataFrame
        Output of identify_market_regimes().

    Returns
    -------
    pd.DataFrame
        One row per regime with: cagr, volatility, sharpe, max_drawdown, n_days.
    """
    # Normalize column names
    nav = nav_series.copy()
    if "nav" in nav.columns and "portfolio_nav" not in nav.columns:
        nav = nav.rename(columns={"nav": "portfolio_nav"})
    if "daily_return" not in nav.columns:
        nav["daily_return"] = nav["portfolio_nav"].pct_change()

    nav["date"] = pd.to_datetime(nav["date"])
    regimes_copy = regimes.copy()
    regimes_copy["date"] = pd.to_datetime(regimes_copy["date"])

    merged = nav.merge(regimes_copy[["date", "regime"]], on="date", how="inner")

    results = []
    for regime_name, group in merged.groupby("regime"):
        if len(group) < 10:
            continue

        returns = group["daily_return"].dropna()
        # Build a mini-NAV for CAGR / max_drawdown calculation
        mini_nav = group[["date", "portfolio_nav", "daily_return"]].reset_index(drop=True)

        metrics = {
            "regime": regime_name,
            "n_days": len(group),
            "pct_of_total": len(group) / len(merged) * 100,
            "cagr": calculate_cagr(mini_nav) if len(mini_nav) > 20 else returns.mean() * 252,
            "volatility": calculate_volatility(returns),
            "sharpe": calculate_sharpe(returns),
            "max_drawdown": calculate_max_drawdown(mini_nav),
            "avg_daily_return": returns.mean(),
            "win_rate": (returns > 0).mean(),
        }
        results.append(metrics)

    df = pd.DataFrame(results).sort_values("n_days", ascending=False)

    logger.info("\nRegime Performance Decomposition:")
    for _, row in df.iterrows():
        logger.info(
            f"  {row['regime']:12s}: CAGR={row['cagr']:+.2%}, "
            f"Sharpe={row['sharpe']:.3f}, MaxDD={row['max_drawdown']:+.2%}, "
            f"Days={row['n_days']}"
        )

    return df.reset_index(drop=True)
