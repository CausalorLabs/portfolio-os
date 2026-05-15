"""
Returns engine — daily, log, cumulative, and rolling return calculations.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_daily_returns(nav: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns from portfolio NAV.

    If nav already has a 'daily_return' column it is recalculated for
    consistency.
    """
    df = nav[["date", "portfolio_nav"]].copy()
    df["daily_return"] = df["portfolio_nav"].pct_change()
    return df


def calculate_log_returns(nav: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log (continuously compounded) returns."""
    df = nav[["date", "portfolio_nav"]].copy()
    df["log_return"] = np.log(df["portfolio_nav"] / df["portfolio_nav"].shift(1))
    return df


def calculate_cumulative_returns(nav: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative return series (base = 0 on first date).

    cumulative_return of 0.25 means +25% since inception.
    """
    df = nav[["date", "portfolio_nav"]].copy()
    df["cumulative_return"] = df["portfolio_nav"] / df["portfolio_nav"].iloc[0] - 1
    return df


def calculate_rolling_returns(nav: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling simple returns over *window* trading days.
    """
    df = nav[["date", "portfolio_nav"]].copy()
    df[f"rolling_{window}d_return"] = (
        df["portfolio_nav"] / df["portfolio_nav"].shift(window) - 1
    )
    return df


def build_returns_table(nav: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a single consolidated returns DataFrame with all return series.

    Columns: date, portfolio_nav, daily_return, log_return,
             cumulative_return, rolling_20d_return, rolling_60d_return
    """
    daily = calculate_daily_returns(nav)
    log_r = calculate_log_returns(nav)
    cum = calculate_cumulative_returns(nav)
    r20 = calculate_rolling_returns(nav, 20)
    r60 = calculate_rolling_returns(nav, 60)

    df = daily.copy()
    df["log_return"] = log_r["log_return"]
    df["cumulative_return"] = cum["cumulative_return"]
    df["rolling_20d_return"] = r20["rolling_20d_return"]
    df["rolling_60d_return"] = r60["rolling_60d_return"]

    logger.info(
        f"Returns table: {len(df)} rows, "
        f"cum return = {df['cumulative_return'].iloc[-1]:+.2%}"
    )
    return df
