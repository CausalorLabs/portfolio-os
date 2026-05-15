"""
Core risk metrics engine — measures quality of portfolio returns.

All annualization uses 252 trading days.
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats


TRADING_DAYS = 252


# ── Individual metric functions ──────────────────────────────────────────────


def calculate_cagr(nav: pd.DataFrame) -> float:
    """Compound Annual Growth Rate."""
    start = nav["portfolio_nav"].iloc[0]
    end = nav["portfolio_nav"].iloc[-1]
    n_days = (nav["date"].iloc[-1] - nav["date"].iloc[0]).days
    if n_days <= 0 or start <= 0:
        return 0.0
    years = n_days / 365.25
    return (end / start) ** (1 / years) - 1


def calculate_annualized_return(daily_returns: pd.Series) -> float:
    """Annualized return from daily simple returns."""
    mean_daily = daily_returns.mean()
    return (1 + mean_daily) ** TRADING_DAYS - 1


def calculate_volatility(daily_returns: pd.Series) -> float:
    """Annualized volatility (standard deviation of daily returns)."""
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def calculate_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate (default 5% for India context).
    """
    ann_return = calculate_annualized_return(daily_returns)
    ann_vol = calculate_volatility(daily_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def calculate_sortino(daily_returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Annualized Sortino ratio — penalizes only downside volatility.
    """
    ann_return = calculate_annualized_return(daily_returns)
    downside = daily_returns[daily_returns < 0]
    if downside.empty:
        return float("inf")
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)
    if downside_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / downside_vol


def calculate_max_drawdown(nav: pd.DataFrame) -> float:
    """Maximum peak-to-trough drawdown (returned as a negative fraction)."""
    prices = nav["portfolio_nav"]
    rolling_peak = prices.cummax()
    drawdown = (prices - rolling_peak) / rolling_peak
    return drawdown.min()


def calculate_calmar(nav: pd.DataFrame, daily_returns: pd.Series) -> float:
    """Calmar ratio = CAGR / |max drawdown|."""
    cagr = calculate_cagr(nav)
    max_dd = calculate_max_drawdown(nav)
    if max_dd == 0:
        return 0.0
    return cagr / abs(max_dd)


def calculate_skewness(daily_returns: pd.Series) -> float:
    """Skewness of daily returns (negative = left tail risk)."""
    return float(sp_stats.skew(daily_returns.dropna()))


def calculate_kurtosis(daily_returns: pd.Series) -> float:
    """Excess kurtosis of daily returns (>0 = fat tails)."""
    return float(sp_stats.kurtosis(daily_returns.dropna()))


# ── Summary report ───────────────────────────────────────────────────────────


def calculate_all_metrics(
    nav: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> dict[str, float]:
    """
    Compute all core risk metrics and return as a dict.
    """
    returns = nav["daily_return"].dropna()

    metrics = {
        "cagr": calculate_cagr(nav),
        "annualized_return": calculate_annualized_return(returns),
        "annualized_volatility": calculate_volatility(returns),
        "sharpe_ratio": calculate_sharpe(returns, risk_free_rate),
        "sortino_ratio": calculate_sortino(returns, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(nav),
        "calmar_ratio": calculate_calmar(nav, returns),
        "skewness": calculate_skewness(returns),
        "kurtosis": calculate_kurtosis(returns),
    }

    _log_metrics(metrics)
    return metrics


def _log_metrics(m: dict[str, float]) -> None:
    """Pretty-print all metrics."""
    logger.info("Core Risk Metrics:")
    logger.info(f"  CAGR              {m['cagr']:>+10.2%}")
    logger.info(f"  Ann. Return       {m['annualized_return']:>+10.2%}")
    logger.info(f"  Ann. Volatility   {m['annualized_volatility']:>10.2%}")
    logger.info(f"  Sharpe Ratio      {m['sharpe_ratio']:>10.3f}")
    logger.info(f"  Sortino Ratio     {m['sortino_ratio']:>10.3f}")
    logger.info(f"  Max Drawdown      {m['max_drawdown']:>+10.2%}")
    logger.info(f"  Calmar Ratio      {m['calmar_ratio']:>10.3f}")
    logger.info(f"  Skewness          {m['skewness']:>10.3f}")
    logger.info(f"  Kurtosis          {m['kurtosis']:>10.3f}")
