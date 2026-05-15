"""
Trend features — moving averages, MA distance, crossovers, trend strength.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_trend_features(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Generate all trend features for each asset."""
    ma = calculate_moving_averages(inr_prices)
    dist = calculate_ma_distance(inr_prices)
    cross = calculate_crossovers(inr_prices)
    strength = calculate_trend_strength(inr_prices)

    result = pd.concat([ma, dist, cross, strength], ignore_index=True)
    logger.info(f"Trend features: {result['feature'].nunique()} features, {len(result)} rows")
    return result


def calculate_moving_averages(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """SMA20, SMA50, SMA200 as features."""
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        price = df["inr_price"]

        for w in [20, 50, 200]:
            sma = price.rolling(w).mean()
            frames.append(_to_long(df["date"], ticker, f"sma_{w}", sma))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_ma_distance(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Distance of price from SMA as fraction: (price - SMA) / SMA."""
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        price = df["inr_price"]

        for w in [20, 50, 200]:
            sma = price.rolling(w).mean()
            dist = (price - sma) / sma
            frames.append(_to_long(df["date"], ticker, f"ma_distance_{w}", dist))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_crossovers(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Binary crossover signals:
      sma20_above_sma50:  1 if SMA20 > SMA50, else 0
      sma50_above_sma200: 1 if SMA50 > SMA200, else 0 (golden cross)
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        price = df["inr_price"]

        sma20 = price.rolling(20).mean()
        sma50 = price.rolling(50).mean()
        sma200 = price.rolling(200).mean()

        cross_20_50 = (sma20 > sma50).astype(float)
        cross_50_200 = (sma50 > sma200).astype(float)

        frames.append(_to_long(df["date"], ticker, "crossover_sma20_sma50", cross_20_50))
        frames.append(_to_long(df["date"], ticker, "crossover_sma50_sma200", cross_50_200))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_trend_strength(
    inr_prices: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Trend strength measured as normalized rolling slope of log prices.
    Positive = uptrend, negative = downtrend.
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        log_price = np.log(df["inr_price"])

        slope = pd.Series(index=df.index, dtype=float)
        for i in range(window, len(df)):
            y = log_price.iloc[i - window : i].values
            x = np.arange(window)
            # Simple linear regression slope
            slope.iloc[i] = np.polyfit(x, y, 1)[0]

        # Annualize: slope per day * 252
        slope = slope * 252
        frames.append(_to_long(df["date"], ticker, f"trend_slope_{window}d", slope))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def _to_long(dates, ticker, feature, values) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates.values,
        "ticker": ticker,
        "feature": feature,
        "value": values.values,
    })
