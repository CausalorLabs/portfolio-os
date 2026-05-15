"""
Factor-like features — proto-factor exposures built from price data.

These are lightweight factor proxies that become inputs for optimization
and ML later. No fundamentals needed at POC stage.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_factor_features(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate factor-like features:
      - momentum_factor:  60D return (cross-sectional rank)
      - low_vol_factor:   inverse of 60D volatility (rank)
      - trend_factor:     SMA50 position signal (rank)
    """
    mom = _momentum_factor(inr_prices)
    low_vol = _low_vol_factor(inr_prices)
    trend = _trend_factor(inr_prices)

    result = pd.concat([mom, low_vol, trend], ignore_index=True)
    logger.info(f"Factor features: {result['feature'].nunique()} features, {len(result)} rows")
    return result


def _momentum_factor(inr_prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Cross-sectional momentum rank (0–1)."""
    ret_data = {}
    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").set_index("date")
        ret_data[ticker] = df["inr_price"] / df["inr_price"].shift(window) - 1

    wide = pd.DataFrame(ret_data)
    ranks = wide.rank(axis=1, pct=True)

    frames = []
    for ticker in ranks.columns:
        frames.append(pd.DataFrame({
            "date": ranks.index,
            "ticker": ticker,
            "feature": "factor_momentum",
            "value": ranks[ticker].values,
        }))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def _low_vol_factor(inr_prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Cross-sectional low-volatility rank (higher = lower vol = better)."""
    vol_data = {}
    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").set_index("date")
        daily_ret = df["inr_price"].pct_change()
        vol_data[ticker] = daily_ret.rolling(window).std() * np.sqrt(252)

    wide = pd.DataFrame(vol_data)
    # Rank ascending then invert: low vol gets high rank
    ranks = wide.rank(axis=1, pct=True, ascending=False)

    frames = []
    for ticker in ranks.columns:
        frames.append(pd.DataFrame({
            "date": ranks.index,
            "ticker": ticker,
            "feature": "factor_low_vol",
            "value": ranks[ticker].values,
        }))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def _trend_factor(inr_prices: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Trend signal: (price - SMA) / SMA, then cross-sectional rank."""
    dist_data = {}
    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").set_index("date")
        sma = df["inr_price"].rolling(window).mean()
        dist_data[ticker] = (df["inr_price"] - sma) / sma

    wide = pd.DataFrame(dist_data)
    ranks = wide.rank(axis=1, pct=True)

    frames = []
    for ticker in ranks.columns:
        frames.append(pd.DataFrame({
            "date": ranks.index,
            "ticker": ticker,
            "feature": "factor_trend",
            "value": ranks[ticker].values,
        }))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])
