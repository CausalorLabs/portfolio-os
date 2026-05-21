"""
Regime feature pipeline — computes daily regime indicators.

Produces:
  vix, vix_zscore, vix_percentile,
  spy_momentum, nifty_momentum,
  breadth_score,
  cross_asset_corr,
  realized_vol, vol_regime_ratio,
  liquidity_stress
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROCESSED = Path("data/processed")
RAW = Path("data/raw")


# ── VIX ──────────────────────────────────────────────────────────────────────


def compute_vix_features(
    vix_prices: pd.DataFrame | None = None,
    zscore_window: int = 252,
) -> pd.DataFrame:
    """
    Compute VIX-based regime features.

    If VIX data is not available, synthesize from portfolio realized vol.

    Returns DataFrame with: date, vix, vix_sma20, vix_zscore, vix_percentile
    """
    if vix_prices is not None and not vix_prices.empty:
        if "date" in vix_prices.columns and "close" in vix_prices.columns:
            df = vix_prices[["date", "close"]].copy()
            df.columns = ["date", "vix"]
        elif "date" in vix_prices.columns and "adj_close" in vix_prices.columns:
            df = vix_prices[["date", "adj_close"]].copy()
            df.columns = ["date", "vix"]
        else:
            df = _synthesize_vix()
    else:
        df = _synthesize_vix()

    df = df.sort_values("date").reset_index(drop=True)
    df["vix_sma20"] = df["vix"].rolling(20, min_periods=10).mean()

    # Z-score: (VIX - rolling mean) / rolling std
    rolling_mean = df["vix"].rolling(zscore_window, min_periods=60).mean()
    rolling_std = df["vix"].rolling(zscore_window, min_periods=60).std()
    df["vix_zscore"] = (df["vix"] - rolling_mean) / rolling_std.replace(0, np.nan)

    # Percentile rank over lookback
    df["vix_percentile"] = (
        df["vix"]
        .rolling(zscore_window, min_periods=60)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    return df[["date", "vix", "vix_sma20", "vix_zscore", "vix_percentile"]].dropna()


def _synthesize_vix() -> pd.DataFrame:
    """Synthesize VIX proxy from SPY realized volatility when VIX data unavailable."""
    logger.warning("VIX data unavailable — synthesizing from SPY realized vol")
    inr_path = PROCESSED / "inr_prices.parquet"
    if not inr_path.exists():
        return pd.DataFrame(columns=["date", "vix"])

    prices = pd.read_parquet(inr_path)
    spy = prices[prices["ticker"] == "SPY"].copy()
    if spy.empty:
        # Fallback to any available equity
        tickers = prices["ticker"].unique()
        if len(tickers) == 0:
            return pd.DataFrame(columns=["date", "vix"])
        spy = prices[prices["ticker"] == tickers[0]].copy()

    spy = spy.sort_values("date")
    spy["daily_return"] = spy["inr_price"].pct_change()
    spy["vix"] = spy["daily_return"].rolling(20, min_periods=10).std() * np.sqrt(252) * 100
    return spy[["date", "vix"]].dropna()


# ── Momentum ─────────────────────────────────────────────────────────────────


def compute_momentum_features(
    inr_prices: pd.DataFrame,
    spy_ticker: str = "SPY",
    nifty_proxy: str = "RELIANCE.NS",
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute market-level momentum indicators.

    Returns DataFrame with: date, spy_momentum, nifty_momentum
    """
    results = []

    for ticker, col_name in [(spy_ticker, "spy_momentum"), (nifty_proxy, "nifty_momentum")]:
        t_data = inr_prices[inr_prices["ticker"] == ticker].copy()
        if t_data.empty:
            continue
        t_data = t_data.sort_values("date")
        t_data[col_name] = t_data["inr_price"].pct_change(window)
        results.append(t_data[["date", col_name]].dropna())

    if not results:
        return pd.DataFrame(columns=["date", "spy_momentum", "nifty_momentum"])

    merged = results[0]
    for r in results[1:]:
        merged = merged.merge(r, on="date", how="outer")

    return merged.sort_values("date").reset_index(drop=True)


# ── Breadth ──────────────────────────────────────────────────────────────────


def compute_breadth_score(
    inr_prices: pd.DataFrame,
    ma_window: int = 200,
    exclude_types: set[str] | None = None,
) -> pd.DataFrame:
    """
    Market breadth: fraction of portfolio tickers above their 200-day MA.

    Returns DataFrame with: date, breadth_score, breadth_advancing
    """
    if exclude_types is None:
        exclude_types = {"fx", "fixed_income"}

    # Load asset master to filter
    master_path = Path("configs/asset_master.csv")
    if master_path.exists():
        master = pd.read_csv(master_path)
        tradable = master[~master["asset_type"].isin(exclude_types)]["ticker"].tolist()
    else:
        tradable = inr_prices["ticker"].unique().tolist()

    prices_wide = (
        inr_prices[inr_prices["ticker"].isin(tradable)]
        .pivot_table(index="date", columns="ticker", values="inr_price", aggfunc="first")
        .sort_index()
        .ffill()
    )

    if prices_wide.empty or len(prices_wide) < ma_window:
        return pd.DataFrame(columns=["date", "breadth_score", "breadth_advancing"])

    sma = prices_wide.rolling(ma_window, min_periods=ma_window).mean()
    above_ma = (prices_wide > sma).astype(float)

    breadth = above_ma.mean(axis=1)
    advancing = above_ma.sum(axis=1)

    result = pd.DataFrame({
        "date": breadth.index,
        "breadth_score": breadth.values,
        "breadth_advancing": advancing.values,
    }).dropna()

    return result.reset_index(drop=True)


# ── Cross-asset correlation ─────────────────────────────────────────────────


def compute_cross_asset_correlation(
    inr_prices: pd.DataFrame,
    window: int = 60,
    exclude_types: set[str] | None = None,
) -> pd.DataFrame:
    """
    Rolling average pairwise correlation across portfolio assets.

    High correlation = systemic stress (diversification failing).

    Returns DataFrame with: date, cross_asset_corr
    """
    if exclude_types is None:
        exclude_types = {"fx", "fixed_income"}

    master_path = Path("configs/asset_master.csv")
    if master_path.exists():
        master = pd.read_csv(master_path)
        tradable = master[~master["asset_type"].isin(exclude_types)]["ticker"].tolist()
    else:
        tradable = inr_prices["ticker"].unique().tolist()

    returns_wide = (
        inr_prices[inr_prices["ticker"].isin(tradable)]
        .pivot_table(index="date", columns="ticker", values="inr_price", aggfunc="first")
        .sort_index()
        .ffill()
        .pct_change()
        .dropna()
    )

    if returns_wide.empty or len(returns_wide.columns) < 3 or len(returns_wide) < window:
        return pd.DataFrame(columns=["date", "cross_asset_corr"])

    # Rolling average pairwise correlation
    avg_corrs = []
    dates = []
    for i in range(window, len(returns_wide)):
        chunk = returns_wide.iloc[i - window : i]
        corr_matrix = chunk.corr()
        # Average off-diagonal correlation
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = corr_matrix.values[mask].mean()
        avg_corrs.append(avg_corr)
        dates.append(returns_wide.index[i])

    return pd.DataFrame({
        "date": dates,
        "cross_asset_corr": avg_corrs,
    }).reset_index(drop=True)


# ── Realized volatility ─────────────────────────────────────────────────────


def compute_realized_vol_features(
    inr_prices: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 252,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Portfolio-level realized volatility and vol regime ratio.

    Returns DataFrame with: date, realized_vol, vol_regime_ratio
    """
    t_data = inr_prices[inr_prices["ticker"] == benchmark].copy()
    if t_data.empty:
        tickers = inr_prices["ticker"].unique()
        if len(tickers) == 0:
            return pd.DataFrame(columns=["date", "realized_vol", "vol_regime_ratio"])
        t_data = inr_prices[inr_prices["ticker"] == tickers[0]].copy()

    t_data = t_data.sort_values("date")
    t_data["daily_return"] = t_data["inr_price"].pct_change()

    t_data["realized_vol"] = (
        t_data["daily_return"].rolling(short_window, min_periods=10).std() * np.sqrt(252)
    )
    long_vol = t_data["daily_return"].rolling(long_window, min_periods=60).std() * np.sqrt(252)
    t_data["vol_regime_ratio"] = t_data["realized_vol"] / long_vol.replace(0, np.nan)

    return t_data[["date", "realized_vol", "vol_regime_ratio"]].dropna().reset_index(drop=True)


# ── Liquidity stress ────────────────────────────────────────────────────────


def compute_liquidity_stress(
    inr_prices: pd.DataFrame,
    atr_window: int = 14,
    spike_multiplier: float = 2.0,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Liquidity stress: ATR expansion relative to median.

    Returns DataFrame with: date, atr_pct, liquidity_stress (0-1 score)
    """
    t_data = inr_prices[inr_prices["ticker"] == benchmark].copy()
    if t_data.empty:
        tickers = inr_prices["ticker"].unique()
        if len(tickers) == 0:
            return pd.DataFrame(columns=["date", "atr_pct", "liquidity_stress"])
        t_data = inr_prices[inr_prices["ticker"] == tickers[0]].copy()

    t_data = t_data.sort_values("date")
    t_data["abs_return"] = t_data["inr_price"].pct_change().abs()
    t_data["atr_pct"] = t_data["abs_return"].rolling(atr_window, min_periods=7).mean()

    # Median ATR over 1 year
    median_atr = t_data["atr_pct"].rolling(252, min_periods=60).median()

    # Stress score: capped at 1.0
    raw_stress = (t_data["atr_pct"] / median_atr.replace(0, np.nan)) / spike_multiplier
    t_data["liquidity_stress"] = raw_stress.clip(0, 1)

    return t_data[["date", "atr_pct", "liquidity_stress"]].dropna().reset_index(drop=True)


# ── Assembler ────────────────────────────────────────────────────────────────


def build_regime_features(
    inr_prices: pd.DataFrame | None = None,
    vix_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the complete daily regime feature table.

    Loads inr_prices from parquet if not provided.

    Returns DataFrame with columns:
        date, vix, vix_sma20, vix_zscore, vix_percentile,
        spy_momentum, nifty_momentum,
        breadth_score, breadth_advancing,
        cross_asset_corr,
        realized_vol, vol_regime_ratio,
        atr_pct, liquidity_stress
    """
    if inr_prices is None:
        path = PROCESSED / "inr_prices.parquet"
        if not path.exists():
            raise FileNotFoundError(f"inr_prices.parquet not found at {path}")
        inr_prices = pd.read_parquet(path)
        inr_prices["date"] = pd.to_datetime(inr_prices["date"])

    logger.info("Building regime features…")

    # Load VIX if available
    if vix_prices is None:
        vix_path = RAW / "^VIX.parquet"
        if vix_path.exists():
            vix_prices = pd.read_parquet(vix_path)
            vix_prices["date"] = pd.to_datetime(vix_prices["date"])
            logger.info("  VIX data loaded from cache")

    vix_df = compute_vix_features(vix_prices)
    mom_df = compute_momentum_features(inr_prices)
    breadth_df = compute_breadth_score(inr_prices)
    corr_df = compute_cross_asset_correlation(inr_prices)
    vol_df = compute_realized_vol_features(inr_prices)
    liq_df = compute_liquidity_stress(inr_prices)

    # Merge all on date
    features = vix_df
    for df in [mom_df, breadth_df, corr_df, vol_df, liq_df]:
        if not df.empty:
            features = features.merge(df, on="date", how="outer")

    features = features.sort_values("date").reset_index(drop=True)

    # Forward-fill gaps (different data sources may have different calendars)
    features = features.ffill()

    n_features = len([c for c in features.columns if c != "date"])
    logger.info(f"  Regime features: {len(features)} days × {n_features} features")

    return features
