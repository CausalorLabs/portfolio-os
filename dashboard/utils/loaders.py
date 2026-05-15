"""
Data loaders — cached functions to load all processed parquet and CSV files.
"""

from pathlib import Path

import pandas as pd
import streamlit as st


PROCESSED = Path("data/processed")
REPORTS = Path("reports")
CONFIGS = Path("configs")
HOLDINGS = Path("data/holdings")


@st.cache_data(ttl=300)
def load_portfolio_nav() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "portfolio_nav.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_inr_prices() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "inr_prices.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_returns() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "returns.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_drawdown_series() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "drawdown_series.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_rolling_analytics() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "rolling_analytics.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_target_weights() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "target_weights.parquet")


@st.cache_data(ttl=300)
def load_signal_scores() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "signal_scores.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_backtest_nav() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "backtest_nav.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_trade_ledger() -> pd.DataFrame:
    path = PROCESSED / "trade_ledger.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_rebalance_trades() -> pd.DataFrame:
    path = PROCESSED / "rebalance_trades.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_portfolio_metrics() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "portfolio_metrics.csv")


@st.cache_data(ttl=300)
def load_backtest_comparison() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "backtest_comparison.csv", index_col=0)


@st.cache_data(ttl=300)
def load_backtest_attribution() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "backtest_attribution.csv")


@st.cache_data(ttl=300)
def load_strategy_comparison() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "strategy_comparison.csv")


@st.cache_data(ttl=300)
def load_portfolio_recommendation() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "portfolio_recommendation.csv")


@st.cache_data(ttl=300)
def load_benchmark_comparison() -> pd.DataFrame:
    path = REPORTS / "benchmark_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_asset_master() -> pd.DataFrame:
    return pd.read_csv(CONFIGS / "asset_master.csv")


@st.cache_data(ttl=300)
def load_holdings() -> pd.DataFrame:
    return pd.read_csv(HOLDINGS / "current_holdings.csv")


@st.cache_data(ttl=300)
def load_fx_attribution() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "fx_attribution.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df
