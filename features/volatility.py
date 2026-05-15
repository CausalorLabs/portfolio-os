"""
Volatility features — realized risk measurement and regime detection.
"""

import numpy as np
import pandas as pd
from loguru import logger


WINDOWS = [20, 60, 120]


def calculate_volatility_features(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Generate all volatility features for each asset."""
    real_vol = calculate_realized_volatility(inr_prices)
    atr = calculate_atr(inr_prices)
    down_vol = calculate_downside_volatility(inr_prices)
    regime = calculate_volatility_regime(inr_prices)

    result = pd.concat([real_vol, atr, down_vol, regime], ignore_index=True)
    logger.info(f"Volatility features: {result['feature'].nunique()} features, {len(result)} rows")
    return result


def calculate_realized_volatility(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Annualized rolling standard deviation of daily returns."""
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        daily_ret = df["inr_price"].pct_change()

        for w in WINDOWS:
            vol = daily_ret.rolling(w).std() * np.sqrt(252)
            frames.append(_to_long(df["date"], ticker, f"volatility_{w}d", vol))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_atr(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Average True Range as a percentage of price.

    Since we only have close prices (not intraday OHLC from the INR-converted
    data), ATR is approximated as rolling mean of absolute daily returns.
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        abs_ret = df["inr_price"].pct_change().abs()

        for w in [14, 20]:
            atr_pct = abs_ret.rolling(w).mean()
            frames.append(_to_long(df["date"], ticker, f"atr_pct_{w}d", atr_pct))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_downside_volatility(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Annualized volatility computed only from negative returns."""
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        daily_ret = df["inr_price"].pct_change()
        downside = daily_ret.clip(upper=0)

        for w in WINDOWS:
            dvol = downside.rolling(w).std() * np.sqrt(252)
            frames.append(_to_long(df["date"], ticker, f"downside_vol_{w}d", dvol))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def calculate_volatility_regime(
    inr_prices: pd.DataFrame,
    lookback: int = 60,
    long_lookback: int = 252,
) -> pd.DataFrame:
    """
    Volatility regime indicator: current vol relative to long-term vol.
    >1 = high-vol regime, <1 = low-vol regime.
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        daily_ret = df["inr_price"].pct_change()

        short_vol = daily_ret.rolling(lookback).std()
        long_vol = daily_ret.rolling(long_lookback).std()
        regime = short_vol / long_vol.replace(0, np.nan)
        frames.append(_to_long(df["date"], ticker, "vol_regime_ratio", regime))

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def _to_long(dates, ticker, feature, values) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates.values,
        "ticker": ticker,
        "feature": feature,
        "value": values.values,
    })
