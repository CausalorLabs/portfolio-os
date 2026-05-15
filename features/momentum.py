"""
Momentum features — absolute, relative, and risk-adjusted momentum.

Measures trend persistence across the asset universe.
"""

import numpy as np
import pandas as pd
from loguru import logger


WINDOWS = [20, 60, 120]


def calculate_momentum_features(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Generate all momentum features for each asset."""
    abs_mom = calculate_absolute_momentum(inr_prices)
    rel_mom = calculate_relative_momentum(inr_prices)
    risk_adj = calculate_risk_adjusted_momentum(inr_prices)

    result = pd.concat([abs_mom, rel_mom, risk_adj], ignore_index=True)
    logger.info(f"Momentum features: {result['feature'].nunique()} features, {len(result)} rows")
    return result


def calculate_absolute_momentum(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Momentum = return over window (positive = trending up)."""
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        price = df["inr_price"]

        for w in WINDOWS:
            mom = price / price.shift(w) - 1
            frames.append(_to_long(df["date"], ticker, f"momentum_{w}d", mom))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_relative_momentum(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional momentum rank (0–1) within the asset universe per date.
    Higher rank = stronger momentum relative to peers.
    """
    frames: list[pd.DataFrame] = []

    for w in WINDOWS:
        # Compute returns per asset
        ret_data = {}
        for ticker, group in inr_prices.groupby("ticker"):
            df = group[["date", "inr_price"]].sort_values("date").set_index("date")
            ret_data[ticker] = (df["inr_price"] / df["inr_price"].shift(w) - 1)

        returns_wide = pd.DataFrame(ret_data)
        # Rank across tickers per date (ascending: worst=0, best=1)
        ranks = returns_wide.rank(axis=1, pct=True)

        for ticker in ranks.columns:
            dates = ranks.index.to_series().reset_index(drop=True)
            values = ranks[ticker].reset_index(drop=True)
            frames.append(_to_long(dates, ticker, f"momentum_rank_{w}d", values))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_risk_adjusted_momentum(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Momentum divided by rolling volatility — sharpe-like momentum signal."""
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        price = df["inr_price"]
        daily_ret = price.pct_change()

        for w in WINDOWS:
            mom = price / price.shift(w) - 1
            vol = daily_ret.rolling(w).std() * np.sqrt(252)
            risk_adj = mom / vol.replace(0, np.nan)
            frames.append(_to_long(df["date"], ticker, f"momentum_riskadjusted_{w}d", risk_adj))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def _to_long(dates, ticker, feature, values) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates.values if hasattr(dates, "values") else dates,
        "ticker": ticker,
        "feature": feature,
        "value": values.values if hasattr(values, "values") else values,
    })
