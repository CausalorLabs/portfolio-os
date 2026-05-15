"""
Portfolio OS — Sprint 1 entry point.

Downloads market data for the test universe, validates quality,
and persists to the local data lake.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from ingestion.yahoo_loader import download_yahoo_data, download_batch
from ingestion.mf_loader import download_mf_data, get_scheme_name
from utils.validators import validate_dataframe


ASSET_MASTER = Path("configs/asset_master.csv")

# SBI Bluechip Fund — Direct Growth (example Indian MF)
TEST_MF_SCHEME = "119598"


def load_asset_master() -> pd.DataFrame:
    """Load the asset master CSV."""
    df = pd.read_csv(ASSET_MASTER)
    logger.info(f"Asset master loaded: {len(df)} assets")
    return df


def run_yahoo_pipeline(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Download and validate all Yahoo-sourced assets."""
    yahoo_tickers = master[master["asset_type"] != "mf"]["ticker"].tolist()
    results: dict[str, pd.DataFrame] = {}

    for ticker in yahoo_tickers:
        df = download_yahoo_data(ticker)
        if not df.empty:
            validate_dataframe(df, ticker)
        results[ticker] = df

    return results


def run_mf_pipeline() -> pd.DataFrame:
    """Download and validate a test mutual fund."""
    name = get_scheme_name(TEST_MF_SCHEME)
    logger.info(f"MF scheme: {name}")

    df = download_mf_data(TEST_MF_SCHEME)
    if not df.empty:
        validate_dataframe(df, f"MF_{TEST_MF_SCHEME}")

    return df


def print_summary(yahoo_data: dict[str, pd.DataFrame], mf_data: pd.DataFrame) -> None:
    """Print a concise summary of all downloaded data."""
    logger.info("=" * 60)
    logger.info("SPRINT 1 — DATA LAKE SUMMARY")
    logger.info("=" * 60)

    for ticker, df in yahoo_data.items():
        if df.empty:
            logger.warning(f"  {ticker:15s} — NO DATA")
        else:
            logger.info(
                f"  {ticker:15s} — {len(df):>6} rows | "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )

    if not mf_data.empty:
        logger.info(
            f"  {'MF_' + TEST_MF_SCHEME:15s} — {len(mf_data):>6} rows | "
            f"{mf_data['date'].min().date()} → {mf_data['date'].max().date()}"
        )

    # List parquet files
    raw_dir = Path("data/raw")
    parquets = list(raw_dir.glob("*.parquet"))
    logger.info(f"\nParquet files in data/raw/: {len(parquets)}")
    for p in sorted(parquets):
        size_kb = p.stat().st_size / 1024
        logger.info(f"  {p.name:30s} {size_kb:>8.1f} KB")


def main() -> None:
    logger.info("Portfolio OS — Sprint 1: Data Pipeline")
    logger.info("-" * 40)

    master = load_asset_master()

    yahoo_data = run_yahoo_pipeline(master)
    mf_data = run_mf_pipeline()

    print_summary(yahoo_data, mf_data)

    logger.info("\nSprint 1 pipeline complete.")


if __name__ == "__main__":
    main()
