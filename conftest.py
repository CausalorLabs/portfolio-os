"""
Root conftest — shared fixtures for the entire test suite.

Provides synthetic market data, portfolio state, and helper factories
so tests run fast without hitting network or disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Deterministic seed ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _seed_rng():
    np.random.seed(42)


# ── Synthetic data factories ─────────────────────────────────────────────────

def _make_price_series(
    n_days: int = 500,
    start: str = "2020-01-01",
    start_price: float = 100.0,
    drift: float = 0.0003,
    vol: float = 0.015,
) -> pd.Series:
    """Generate a synthetic daily price series (geometric random walk)."""
    dates = pd.bdate_range(start, periods=n_days)
    log_returns = np.random.normal(drift, vol, n_days)
    prices = start_price * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=dates, name="price")


# ── Core fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def asset_master() -> pd.DataFrame:
    """Minimal asset master for testing."""
    return pd.DataFrame({
        "ticker": ["AAPL", "SPY", "RELIANCE.NS", "INFY.NS", "USDINR=X"],
        "asset_type": ["equity", "etf", "equity", "equity", "fx"],
        "country": ["US", "US", "IN", "IN", "GLOBAL"],
        "currency": ["USD", "USD", "INR", "INR", "INR"],
    })


@pytest.fixture()
def country_map(asset_master) -> dict[str, str]:
    return dict(zip(asset_master["ticker"], asset_master["country"]))


@pytest.fixture()
def wide_prices() -> pd.DataFrame:
    """Wide-format daily prices (4 assets, ~500 days)."""
    tickers = ["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"]
    dfs = {}
    for i, t in enumerate(tickers):
        dfs[t] = _make_price_series(
            n_days=500,
            start_price=100 + i * 50,
            drift=0.0003 + i * 0.00005,
            vol=0.012 + i * 0.002,
        )
    df = pd.DataFrame(dfs)
    df.index.name = "date"
    return df


@pytest.fixture()
def wide_returns(wide_prices) -> pd.DataFrame:
    """Daily returns derived from wide_prices."""
    return wide_prices.pct_change().dropna()


@pytest.fixture()
def inr_prices() -> pd.DataFrame:
    """Long-format INR prices (mimics fx/converter output)."""
    tickers = ["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"]
    dates = pd.bdate_range("2020-01-01", periods=500)
    rows = []
    for t in tickers:
        prices = _make_price_series(n_days=500, start_price=1000)
        for d, p in zip(dates, prices):
            rows.append({"date": d, "ticker": t, "inr_price": p})
    return pd.DataFrame(rows)


@pytest.fixture()
def nav_df(wide_prices) -> pd.DataFrame:
    """NAV DataFrame compatible with analytics functions."""
    cumret = (1 + wide_prices.pct_change().fillna(0)).cumprod()
    nav = cumret.mean(axis=1) * 1_000_000
    return pd.DataFrame({
        "date": cumret.index,
        "portfolio_nav": nav.values,
    })


@pytest.fixture()
def daily_returns(nav_df) -> pd.Series:
    """Daily portfolio returns series."""
    return nav_df["portfolio_nav"].pct_change().dropna()


@pytest.fixture()
def holdings() -> pd.DataFrame:
    """Sample holdings DataFrame."""
    return pd.DataFrame({
        "ticker": ["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"],
        "quantity": [10, 5, 15, 25],
        "avg_cost": [180.0, 420.0, 2500.0, 1400.0],
        "currency": ["USD", "USD", "INR", "INR"],
    })


@pytest.fixture()
def equal_weight_strategy():
    """Simple equal-weight strategy_fn for backtesting."""
    def strategy(returns: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
        n = len(tickers)
        return {t: 1.0 / n for t in tickers}
    return strategy


@pytest.fixture()
def feature_store(inr_prices) -> pd.DataFrame:
    """Minimal feature store for testing signal functions."""
    dates = inr_prices["date"].unique()[:252]
    tickers = ["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "momentum_20d": np.random.normal(0, 0.05),
                "momentum_60d": np.random.normal(0, 0.08),
                "rolling_20d_vol": abs(np.random.normal(0.015, 0.005)),
                "rolling_60d_vol": abs(np.random.normal(0.018, 0.005)),
                "trend_sma_ratio_50": 1 + np.random.normal(0, 0.03),
                "trend_sma_ratio_200": 1 + np.random.normal(0, 0.05),
                "mean_reversion_zscore_20d": np.random.normal(0, 1),
                "return_1d": np.random.normal(0.0003, 0.015),
                "return_5d": np.random.normal(0.0015, 0.03),
                "return_20d": np.random.normal(0.006, 0.06),
            })
    return pd.DataFrame(rows)
