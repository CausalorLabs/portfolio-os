"""
FX conversion engine — converts all asset prices to INR.

Maintains both native and INR-denominated prices for auditability.
"""

import pandas as pd
from loguru import logger

from fx.fx_loader import get_fx_series


def convert_prices_to_inr(
    asset_prices: dict[str, pd.DataFrame],
    asset_master: pd.DataFrame,
    fx_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert asset prices to INR using FX rates.

    Parameters
    ----------
    asset_prices : dict[str, pd.DataFrame]
        Mapping of ticker -> price DataFrame (must have 'date' and 'adj_close').
    asset_master : pd.DataFrame
        Asset master with 'ticker' and 'currency' columns.
    fx_series : pd.DataFrame, optional
        Pre-loaded FX series. If None, loads automatically.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, native_price, inr_price, fx_rate, fx_source_date, currency
    """
    if fx_series is None:
        fx_series = get_fx_series()

    if fx_series.empty:
        raise ValueError("FX series is empty — cannot convert prices")

    currency_map = dict(zip(asset_master["ticker"], asset_master["currency"]))

    frames: list[pd.DataFrame] = []

    for ticker, prices_df in asset_prices.items():
        if prices_df.empty:
            logger.warning(f"Skipping {ticker}: empty price data")
            continue

        currency = currency_map.get(ticker, "INR")

        # FX pairs are not portfolio assets — skip them
        asset_type = asset_master.loc[
            asset_master["ticker"] == ticker, "asset_type"
        ]
        if not asset_type.empty and asset_type.iloc[0] == "fx":
            continue

        # Fixed-income assets are always INR and don't need FX conversion
        if not asset_type.empty and asset_type.iloc[0] == "fixed_income":
            currency = "INR"

        # Metal proxy prices from Yahoo are in USD (commodity futures)
        if not asset_type.empty and asset_type.iloc[0] == "metal":
            currency = "USD"

        converted = _convert_single_asset(
            ticker, prices_df, currency, fx_series
        )
        frames.append(converted)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    n_assets = result["ticker"].nunique()
    logger.info(f"INR conversion complete: {n_assets} assets, {len(result)} total rows")

    return result


def _convert_single_asset(
    ticker: str,
    prices_df: pd.DataFrame,
    currency: str,
    fx_series: pd.DataFrame,
) -> pd.DataFrame:
    """Convert one asset's prices to INR."""
    df = prices_df[["date", "adj_close"]].copy()
    df = df.rename(columns={"adj_close": "native_price"})
    df["ticker"] = ticker
    df["currency"] = currency

    if currency == "USD":
        # Merge FX rates by date
        df = df.merge(
            fx_series[["date", "fx_rate", "fx_source_date"]],
            on="date",
            how="left",
        )

        # For dates without FX (shouldn't happen after ffill, but guard)
        missing_fx = df["fx_rate"].isnull().sum()
        if missing_fx > 0:
            logger.warning(f"{ticker}: {missing_fx} dates missing FX rate — forward-filling")
            df["fx_rate"] = df["fx_rate"].ffill()
            df["fx_source_date"] = df["fx_source_date"].ffill()

        df["inr_price"] = df["native_price"] * df["fx_rate"]

    elif currency == "INR":
        df["fx_rate"] = 1.0
        df["fx_source_date"] = df["date"]
        df["inr_price"] = df["native_price"]

    else:
        raise ValueError(f"Unsupported currency '{currency}' for ticker {ticker}")

    # Drop rows where conversion failed
    df = df.dropna(subset=["inr_price"])

    return df[["date", "ticker", "native_price", "inr_price", "fx_rate", "fx_source_date", "currency"]]
