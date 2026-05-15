"""
Feature store — central repository for all computed features.

Stores features in long format (date, ticker, feature, value) as parquet.
Provides access patterns for optimization, ML, and analytics.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from features.returns import calculate_returns_features
from features.momentum import calculate_momentum_features
from features.volatility import calculate_volatility_features
from features.trend import calculate_trend_features
from features.mean_reversion import calculate_mean_reversion_features
from features.factor_features import calculate_factor_features


FEATURE_STORE_PATH = Path("data/processed/features.parquet")


def build_feature_store(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features and assemble into a single long-format dataframe.

    Schema: date | ticker | feature | value
    """
    logger.info("Building feature store...")

    modules = [
        ("returns", calculate_returns_features),
        ("momentum", calculate_momentum_features),
        ("volatility", calculate_volatility_features),
        ("trend", calculate_trend_features),
        ("mean_reversion", calculate_mean_reversion_features),
        ("factor", calculate_factor_features),
    ]

    frames: list[pd.DataFrame] = []
    for name, fn in modules:
        logger.info(f"  Computing {name} features...")
        df = fn(inr_prices)
        frames.append(df)

    store = pd.concat(frames, ignore_index=True)
    store = store.sort_values(["date", "ticker", "feature"]).reset_index(drop=True)

    _log_summary(store)
    return store


def save_feature_store(store: pd.DataFrame, path: Path = FEATURE_STORE_PATH) -> Path:
    """Persist the feature store to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(path, index=False, engine="pyarrow")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Feature store saved → {path}  ({size_mb:.2f} MB)")
    return path


def load_feature_store(path: Path = FEATURE_STORE_PATH) -> pd.DataFrame:
    """Load feature store from parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Feature store not found at {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Feature store loaded: {len(df)} rows, {df['feature'].nunique()} features")
    return df


def get_features_wide(
    store: pd.DataFrame,
    ticker: str | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pivot feature store from long to wide format for analysis.

    Returns: date × feature matrix (optionally filtered by ticker/features).
    """
    df = store.copy()
    if ticker:
        df = df[df["ticker"] == ticker]
    if features:
        df = df[df["feature"].isin(features)]

    wide = df.pivot_table(index="date", columns="feature", values="value", aggfunc="first")
    return wide


def get_latest_features(store: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent feature values for all tickers."""
    latest_date = store["date"].max()
    latest = store[store["date"] == latest_date].copy()
    return latest.pivot_table(index="ticker", columns="feature", values="value", aggfunc="first")


def _log_summary(store: pd.DataFrame) -> None:
    """Log feature store summary."""
    n_features = store["feature"].nunique()
    n_tickers = store["ticker"].nunique()
    n_dates = store["date"].nunique()
    logger.info(f"Feature store complete:")
    logger.info(f"  {n_features} features × {n_tickers} tickers × {n_dates} dates")
    logger.info(f"  {len(store):,} total rows")

    # Feature list
    features = sorted(store["feature"].unique())
    logger.info(f"  Features: {', '.join(features[:10])}{'...' if len(features) > 10 else ''}")
