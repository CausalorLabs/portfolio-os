"""
Yahoo Finance data loader for equities, ETFs, and FX.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger


RAW_DIR = Path("data/raw")


def download_yahoo_data(
    ticker: str,
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
    save: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance, normalize schema, and optionally
    persist as parquet.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "AAPL", "RELIANCE.NS", "USDINR=X").
    start_date : str
        ISO date string for the download window start.
    end_date : str
        ISO date string for the download window end.
    save : bool
        If True, save the cleaned dataframe to data/raw/<ticker>.parquet.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV dataframe with columns:
        date, open, high, low, close, adj_close, volume, ticker
    """
    logger.info(f"Downloading {ticker} from {start_date} to {end_date}")

    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if raw.empty:
        logger.warning(f"No data returned for {ticker}")
        return pd.DataFrame()

    df = _normalize(raw, ticker)
    df = _clean(df)

    logger.info(f"{ticker}: {len(df)} rows, {df['date'].min()} → {df['date'].max()}")

    if save:
        _save_parquet(df, ticker)

    return df


def download_batch(
    tickers: list[str],
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
) -> dict[str, pd.DataFrame]:
    """Download data for a list of tickers and return a dict of DataFrames."""
    results: dict[str, pd.DataFrame] = {}
    for t in tickers:
        results[t] = download_yahoo_data(t, start_date, end_date, save=True)
    return results


# ── internal helpers ─────────────────────────────────────────────────────────


def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten multi-index columns and standardize column names."""
    df = raw.copy()

    # yfinance may return MultiIndex columns for single ticker downloads
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # If adj_close is missing, fall back to close
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df["ticker"] = ticker

    expected = ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]
    return df[[c for c in expected if c in df.columns]]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, sort, and coerce types."""
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _save_parquet(df: pd.DataFrame, ticker: str) -> Path:
    """Write dataframe to parquet in data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = ticker.replace("=", "_").replace("/", "_")
    path = RAW_DIR / f"{safe_name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Saved → {path}")
    return path
