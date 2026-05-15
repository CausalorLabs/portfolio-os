"""
Returns feature engine — multi-horizon return features per asset.

All features use ONLY past data (no lookahead).
"""

import pandas as pd
from loguru import logger


WINDOWS = [5, 20, 60, 120]


def calculate_returns_features(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate return features for each asset across multiple horizons.

    Parameters
    ----------
    inr_prices : pd.DataFrame
        Must have columns: date, ticker, inr_price.

    Returns
    -------
    pd.DataFrame
        Long-format: date, ticker, feature, value
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group[["date", "inr_price"]].sort_values("date").copy()
        price = df["inr_price"]

        # Daily return
        frames.append(_to_long(df["date"], ticker, "return_1d", price.pct_change()))

        # Rolling returns at multiple horizons
        for w in WINDOWS:
            ret = price / price.shift(w) - 1
            frames.append(_to_long(df["date"], ticker, f"return_{w}d", ret))

        # Cumulative return from inception
        cum = price / price.iloc[0] - 1
        frames.append(_to_long(df["date"], ticker, "return_cumulative", cum))

    result = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    logger.info(f"Returns features: {result['feature'].nunique()} features, {len(result)} rows")
    return result


def _to_long(
    dates: pd.Series,
    ticker: str,
    feature: str,
    values: pd.Series,
) -> pd.DataFrame:
    """Convert a series to long-format feature rows."""
    return pd.DataFrame({
        "date": dates.values,
        "ticker": ticker,
        "feature": feature,
        "value": values.values,
    })
