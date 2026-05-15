"""
Benchmark comparison engine — compare portfolio against passive alternatives.
"""

import numpy as np
import pandas as pd
from loguru import logger

from analytics.metrics import (
    TRADING_DAYS,
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
)


def compare_against_benchmark(
    portfolio_nav: pd.DataFrame,
    benchmark_nav: pd.DataFrame,
    benchmark_name: str = "Benchmark",
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Compare portfolio metrics against a benchmark.

    Both inputs must have 'date' and 'portfolio_nav' columns.
    Returns a side-by-side metrics comparison.
    """
    # Align dates
    merged = portfolio_nav[["date", "portfolio_nav"]].merge(
        benchmark_nav[["date", "portfolio_nav"]],
        on="date",
        how="inner",
        suffixes=("_portfolio", "_benchmark"),
    )

    if merged.empty:
        logger.warning("No overlapping dates between portfolio and benchmark")
        return pd.DataFrame()

    # Build aligned NAV frames
    port = pd.DataFrame({
        "date": merged["date"],
        "portfolio_nav": merged["portfolio_nav_portfolio"],
    })
    bench = pd.DataFrame({
        "date": merged["date"],
        "portfolio_nav": merged["portfolio_nav_benchmark"],
    })

    port["daily_return"] = port["portfolio_nav"].pct_change()
    bench["daily_return"] = bench["portfolio_nav"].pct_change()

    pr = port["daily_return"].dropna()
    br = bench["daily_return"].dropna()

    metrics = {
        "metric": [
            "CAGR",
            "Ann. Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Correlation",
            "Tracking Error",
            "Information Ratio",
        ],
        "Portfolio": [
            calculate_cagr(port),
            calculate_volatility(pr),
            calculate_sharpe(pr, risk_free_rate),
            calculate_sortino(pr, risk_free_rate),
            calculate_max_drawdown(port),
            np.nan,
            np.nan,
            np.nan,
        ],
        benchmark_name: [
            calculate_cagr(bench),
            calculate_volatility(br),
            calculate_sharpe(br, risk_free_rate),
            calculate_sortino(br, risk_free_rate),
            calculate_max_drawdown(bench),
            np.nan,
            np.nan,
            np.nan,
        ],
    }

    corr = pr.corr(br)
    te = calculate_tracking_error(pr, br)
    ir = calculate_information_ratio(pr, br)

    # Fill correlation / TE / IR only in portfolio column (they are relative)
    metrics["Portfolio"][5] = corr
    metrics["Portfolio"][6] = te
    metrics["Portfolio"][7] = ir
    metrics[benchmark_name][5] = corr
    metrics[benchmark_name][6] = te
    metrics[benchmark_name][7] = ir

    result = pd.DataFrame(metrics)
    _log_comparison(result, benchmark_name)

    return result


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Annualized tracking error (std of excess returns)."""
    excess = portfolio_returns - benchmark_returns
    return float(excess.std() * np.sqrt(TRADING_DAYS))


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Information ratio = annualized excess return / tracking error."""
    excess = portfolio_returns - benchmark_returns
    te = excess.std() * np.sqrt(TRADING_DAYS)
    if te == 0:
        return 0.0
    ann_excess = excess.mean() * TRADING_DAYS
    return float(ann_excess / te)


def build_benchmark_nav(
    asset_prices: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Build a simple NAV series for a benchmark asset from inr_prices data.

    Normalizes to start at the same value as the portfolio's first date.
    """
    bench = asset_prices[asset_prices["ticker"] == ticker][["date", "inr_price"]].copy()
    bench = bench.sort_values("date").reset_index(drop=True)

    if bench.empty:
        logger.warning(f"No INR price data for benchmark {ticker}")
        return pd.DataFrame(columns=["date", "portfolio_nav"])

    bench = bench.rename(columns={"inr_price": "portfolio_nav"})
    return bench


def _log_comparison(df: pd.DataFrame, benchmark_name: str) -> None:
    """Log the comparison table."""
    logger.info(f"Portfolio vs {benchmark_name}:")
    for _, row in df.iterrows():
        metric = row["metric"]
        pv = row["Portfolio"]
        bv = row[benchmark_name]

        if metric in ("CAGR", "Ann. Volatility", "Max Drawdown", "Tracking Error"):
            logger.info(f"  {metric:20s}  {pv:>+8.2%}  vs  {bv:>+8.2%}")
        elif metric == "Correlation":
            logger.info(f"  {metric:20s}  {pv:>8.3f}")
        else:
            logger.info(f"  {metric:20s}  {pv:>8.3f}  vs  {bv:>8.3f}")
