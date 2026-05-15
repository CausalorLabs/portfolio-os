"""
FX data loader — fetches and manages USDINR historical rates.

Reuses the Yahoo loader for download, then provides FX-specific
access patterns: daily rate lookup, date-range slicing, and
forward-fill with audit tracking.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from ingestion.yahoo_loader import download_yahoo_data


RAW_DIR = Path("data/raw")
DEFAULT_PAIR = "USDINR=X"


def load_fx_data(
    pair: str = DEFAULT_PAIR,
    start_date: str = "2018-01-01",
    end_date: str = "2026-01-01",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load FX rate history.  Downloads from Yahoo if no cached parquet exists
    (or if force_download is True).

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, adj_close, ticker
    """
    safe_name = pair.replace("=", "_").replace("/", "_")
    parquet_path = RAW_DIR / f"{safe_name}.parquet"

    if parquet_path.exists() and not force_download:
        logger.info(f"Loading cached FX data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    logger.info(f"Downloading FX pair {pair}")
    df = download_yahoo_data(pair, start_date=start_date, end_date=end_date, save=True)
    return df


def get_fx_series(
    pair: str = DEFAULT_PAIR,
    start_date: str = "2018-01-01",
    end_date: str = "2026-01-01",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Return a clean (date, fx_rate, fx_source_date) series with controlled
    forward-fill for weekends/holidays.

    fx_source_date tracks the actual trading date from which the rate
    originates, ensuring auditability when rates are forward-filled.
    """
    raw = load_fx_data(pair, start_date, end_date, force_download)

    if raw.empty:
        return pd.DataFrame(columns=["date", "fx_rate", "fx_source_date"])

    fx = raw[["date", "close"]].copy()
    fx = fx.rename(columns={"close": "fx_rate"})
    fx["fx_source_date"] = fx["date"]

    # Build a continuous calendar and forward-fill (max 5 days for long weekends)
    full_range = pd.date_range(fx["date"].min(), fx["date"].max(), freq="D")
    fx = fx.set_index("date").reindex(full_range)
    fx.index.name = "date"

    fx["fx_rate"] = fx["fx_rate"].ffill(limit=5)
    fx["fx_source_date"] = fx["fx_source_date"].ffill(limit=5)

    fx = fx.dropna(subset=["fx_rate"]).reset_index()

    logger.info(
        f"FX series: {len(fx)} days, "
        f"{fx['date'].min().date()} → {fx['date'].max().date()}"
    )
    return fx


def get_fx_rate_on_date(fx_series: pd.DataFrame, target_date: str | pd.Timestamp) -> float:
    """Look up the FX rate for a specific date."""
    target = pd.Timestamp(target_date)
    match = fx_series.loc[fx_series["date"] == target, "fx_rate"]
    if match.empty:
        # Find nearest prior date
        prior = fx_series.loc[fx_series["date"] <= target]
        if prior.empty:
            raise ValueError(f"No FX rate available on or before {target_date}")
        return float(prior.iloc[-1]["fx_rate"])
    return float(match.iloc[0])
