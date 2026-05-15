"""
Rolling analytics engine — time-varying diagnostics for volatility,
Sharpe, beta, and drawdown.
"""

import numpy as np
import pandas as pd
from loguru import logger

from analytics.metrics import TRADING_DAYS


def calculate_rolling_volatility(
    daily_returns: pd.Series,
    dates: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """Annualized rolling volatility."""
    vol = daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return pd.DataFrame({"date": dates, f"rolling_{window}d_vol": vol})


def calculate_rolling_sharpe(
    daily_returns: pd.Series,
    dates: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """Annualized rolling Sharpe ratio."""
    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std()

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    sharpe = (rolling_mean - daily_rf) / rolling_std * np.sqrt(TRADING_DAYS)

    return pd.DataFrame({"date": dates, f"rolling_{window}d_sharpe": sharpe})


def calculate_rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    dates: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling beta relative to a benchmark."""
    cov = portfolio_returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var()
    beta = cov / var

    return pd.DataFrame({"date": dates, f"rolling_{window}d_beta": beta})


def calculate_rolling_drawdown(
    nav_series: pd.Series,
    dates: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling window max drawdown."""
    dd = pd.Series(index=nav_series.index, dtype=float)

    for i in range(window, len(nav_series)):
        window_slice = nav_series.iloc[i - window : i + 1]
        peak = window_slice.cummax()
        drawdown = (window_slice - peak) / peak
        dd.iloc[i] = drawdown.min()

    return pd.DataFrame({"date": dates, f"rolling_{window}d_max_dd": dd})


def build_rolling_table(
    nav: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Produce a consolidated rolling analytics DataFrame.

    Parameters
    ----------
    nav : pd.DataFrame
        Must have columns: date, portfolio_nav, daily_return.
    benchmark_returns : pd.Series, optional
        Daily returns of a benchmark (aligned to same dates).
    """
    dates = nav["date"]
    returns = nav["daily_return"]
    nav_series = nav["portfolio_nav"]

    vol20 = calculate_rolling_volatility(returns, dates, 20)
    vol60 = calculate_rolling_volatility(returns, dates, 60)
    sharpe60 = calculate_rolling_sharpe(returns, dates, 60)
    dd60 = calculate_rolling_drawdown(nav_series, dates, 60)

    df = vol20.copy()
    df["rolling_60d_vol"] = vol60["rolling_60d_vol"]
    df["rolling_60d_sharpe"] = sharpe60["rolling_60d_sharpe"]
    df["rolling_60d_max_dd"] = dd60["rolling_60d_max_dd"]

    if benchmark_returns is not None:
        beta60 = calculate_rolling_beta(returns, benchmark_returns, dates, 60)
        df["rolling_60d_beta"] = beta60["rolling_60d_beta"]

    valid = df.dropna(subset=["rolling_20d_vol"])
    logger.info(
        f"Rolling analytics: {len(valid)} valid rows, "
        f"latest vol={valid['rolling_20d_vol'].iloc[-1]:.2%}, "
        f"sharpe={valid['rolling_60d_sharpe'].iloc[-1]:.3f}"
    )

    return df
