"""
Mean reversion features — RSI, Bollinger Bands, price z-score.

Identifies temporary overextension in either direction.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_mean_reversion_features(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Generate all mean-reversion features for each asset."""
    rsi = calculate_rsi(inr_prices)
    boll = calculate_bollinger_bands(inr_prices)
    zscore = calculate_price_zscore(inr_prices)

    result = pd.concat([rsi, boll, zscore], ignore_index=True)
    logger.info(f"Mean reversion features: {result['feature'].nunique()} features, {len(result)} rows")
    return result


def calculate_rsi(inr_prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (Wilder's smoothing).
    RSI > 70 → overbought, RSI < 30 → oversold.
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        delta = df["inr_price"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's exponential moving average
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        frames.append(_to_long(df["date"], ticker, f"rsi_{period}", rsi))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_bollinger_bands(
    inr_prices: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Band features:
      bb_zscore:      (price - SMA) / (num_std * rolling_std)
      bb_width:       band width as fraction of SMA
      bb_pct_b:       %B — position within bands (0=lower, 1=upper)
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        price = df["inr_price"]

        sma = price.rolling(window).mean()
        std = price.rolling(window).std()

        upper = sma + num_std * std
        lower = sma - num_std * std

        # Z-score
        zscore = (price - sma) / (std.replace(0, np.nan))
        frames.append(_to_long(df["date"], ticker, "bb_zscore", zscore))

        # Band width
        width = (upper - lower) / sma
        frames.append(_to_long(df["date"], ticker, "bb_width", width))

        # %B
        pct_b = (price - lower) / (upper - lower).replace(0, np.nan)
        frames.append(_to_long(df["date"], ticker, "bb_pct_b", pct_b))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_price_zscore(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price z-score: (price - rolling_mean) / rolling_std
    Measures how many standard deviations price is from recent mean.
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        price = df["inr_price"]

        for w in [20, 60]:
            mean = price.rolling(w).mean()
            std = price.rolling(w).std()
            z = (price - mean) / std.replace(0, np.nan)
            frames.append(_to_long(df["date"], ticker, f"price_zscore_{w}d", z))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def _to_long(dates, ticker, feature, values) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates.values,
        "ticker": ticker,
        "feature": feature,
        "value": values.values,
    })
