"""
Fixed-income data loader — generates synthetic daily prices for
instruments with administered/fixed interest rates (EPF, PPF, FD, NPS, SGB).

These assets don't have market data; instead we model daily accrual
from the annual rate, treating avg_buy_price × quantity as the principal.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


RAW_DIR = Path("data/raw")


def generate_fixed_income_prices(
    ticker: str,
    annual_rate: float,
    start_date: str | pd.Timestamp = "2020-01-01",
    end_date: str | pd.Timestamp | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic daily prices for a fixed-income instrument.

    The price starts at 100.0 and compounds daily at the given annual rate.
    This gives us a time series compatible with the rest of the pipeline.

    Parameters
    ----------
    ticker : str
        Instrument identifier (e.g. "EPF", "PPF", "FD_SBI").
    annual_rate : float
        Annual interest rate as a percentage (e.g. 8.25 for 8.25%).
    start_date : str or Timestamp
        First date in the series.
    end_date : str or Timestamp or None
        Last date (defaults to today).
    save : bool
        If True, persist to data/raw/<ticker>.parquet.

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, adj_close, volume, ticker
        (OHLCV format for pipeline compatibility)
    """
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()

    dates = pd.bdate_range(start=start_date, end=end_date)
    if len(dates) == 0:
        logger.warning(f"{ticker}: no business days in range")
        return pd.DataFrame()

    daily_rate = (1 + annual_rate / 100) ** (1 / 252) - 1
    prices = 100.0 * np.cumprod(np.ones(len(dates)) * (1 + daily_rate))

    df = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "adj_close": prices,
        "volume": 0,
        "ticker": ticker,
    })

    logger.info(
        f"{ticker}: {len(df)} rows @ {annual_rate}% p.a. | "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )

    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / f"{ticker}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info(f"Saved → {path}")

    return df


def generate_metal_proxy_prices(
    ticker: str,
    yahoo_ticker: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download commodity prices from Yahoo and relabel for physical metal tracking.

    Uses Yahoo commodity futures as price proxy:
      GOLD_PHYS → GC=F (gold futures, USD/oz → converted to INR/gram in FX stage)
      SILVER_PHYS → SI=F (silver futures, USD/oz → converted to INR/gram in FX stage)

    Parameters
    ----------
    ticker : str
        Our internal ticker (e.g. "GOLD_PHYS").
    yahoo_ticker : str
        Yahoo Finance commodity ticker (e.g. "GC=F").
    save : bool
        If True, persist to data/raw/<ticker>.parquet.

    Returns
    -------
    pd.DataFrame
        Standard OHLCV format with our internal ticker.
    """
    from ingestion.yahoo_loader import download_yahoo_data

    df = download_yahoo_data(yahoo_ticker)
    if df.empty:
        logger.warning(f"{ticker}: no data from Yahoo for {yahoo_ticker}")
        return pd.DataFrame()

    # Relabel with our internal ticker
    df["ticker"] = ticker

    logger.info(
        f"{ticker} (via {yahoo_ticker}): {len(df)} rows | "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )

    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / f"{ticker}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info(f"Saved → {path}")

    return df


# Mapping from our metal tickers to Yahoo commodity tickers
METAL_YAHOO_MAP = {
    "GOLD_PHYS": "GC=F",
    "SILVER_PHYS": "SI=F",
}
