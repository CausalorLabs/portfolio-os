"""
Holdings loader — reads portfolio positions from CSV.
"""

from pathlib import Path

import pandas as pd
from loguru import logger


DEFAULT_HOLDINGS_PATH = Path("data/holdings/current_holdings.csv")

REQUIRED_COLUMNS = {"ticker", "quantity", "avg_buy_price", "currency", "asset_type"}
SUPPORTED_CURRENCIES = {"USD", "INR"}


def load_holdings(filepath: str | Path = DEFAULT_HOLDINGS_PATH) -> pd.DataFrame:
    """
    Load portfolio holdings from CSV and validate.

    Expected columns: ticker, quantity, avg_buy_price, currency, asset_type.

    Returns
    -------
    pd.DataFrame
        Validated holdings with cost_basis column added.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Holdings file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Holdings CSV missing columns: {missing}")

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["avg_buy_price"] = pd.to_numeric(df["avg_buy_price"], errors="coerce")
    df["currency"] = df["currency"].str.upper().str.strip()
    df["ticker"] = df["ticker"].str.strip()

    _validate(df)

    # Add cost basis in native currency
    df["cost_basis"] = df["quantity"] * df["avg_buy_price"]

    logger.info(f"Holdings loaded: {len(df)} positions, {df['currency'].nunique()} currencies")
    logger.info(f"  Tickers: {', '.join(df['ticker'].tolist())}")

    return df


def _validate(df: pd.DataFrame) -> None:
    """Run validation checks on holdings; raise on critical issues."""
    # Duplicate tickers
    dups = df[df.duplicated(subset=["ticker"], keep=False)]
    if not dups.empty:
        raise ValueError(f"Duplicate tickers in holdings: {dups['ticker'].tolist()}")

    # Negative quantities
    neg = df[df["quantity"] <= 0]
    if not neg.empty:
        raise ValueError(f"Non-positive quantities for: {neg['ticker'].tolist()}")

    # Unsupported currencies
    unsupported = set(df["currency"]) - SUPPORTED_CURRENCIES
    if unsupported:
        raise ValueError(f"Unsupported currencies: {unsupported}")

    # Missing values in key fields
    if df[["ticker", "quantity", "avg_buy_price", "currency"]].isnull().any().any():
        raise ValueError("Holdings contain null values in required fields")
