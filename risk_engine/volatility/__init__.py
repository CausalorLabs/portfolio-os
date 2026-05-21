"""
Risk Engine — Dynamic Volatility Engine.

Estimates current market risk state using:
  - EWMA volatility (multi-horizon: 20d/60d/120d)
  - Realized volatility (5d/20d/60d)
  - Volatility regime classification (normal/elevated/panic)
  - Percentile ranking

Output: date | ticker | ewma_vol | realized_vol | vol_regime | vol_percentile
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/risk_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ── EWMA Volatility ─────────────────────────────────────────────────────────


def compute_ewma_volatility(
    returns: pd.DataFrame,
    span: int = 60,
) -> pd.DataFrame:
    """
    Compute EWMA volatility per ticker (annualized).

    Parameters
    ----------
    returns : Wide-format daily returns (columns = tickers)
    span : EWMA span (half-life in days)

    Returns
    -------
    DataFrame: same shape as returns, annualized EWMA vol
    """
    ewma_var = returns.ewm(span=span, min_periods=max(10, span // 3)).var()
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)
    return ewma_vol


def compute_multi_horizon_ewma(
    returns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute EWMA vol at short/medium/long horizons."""
    cfg = _load_config().get("volatility", {}).get("ewma", {})
    short = cfg.get("short_span", 20)
    medium = cfg.get("medium_span", 60)
    long_ = cfg.get("long_span", 120)

    return {
        f"ewma_vol_{short}d": compute_ewma_volatility(returns, short),
        f"ewma_vol_{medium}d": compute_ewma_volatility(returns, medium),
        f"ewma_vol_{long_}d": compute_ewma_volatility(returns, long_),
    }


# ── Realized Volatility ─────────────────────────────────────────────────────


def compute_realized_volatility(
    returns: pd.DataFrame,
    windows: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute rolling realized volatility at multiple horizons.

    Returns dict of DataFrames keyed by 'realized_vol_{window}d'.
    """
    cfg = _load_config().get("volatility", {}).get("realized", {})
    if windows is None:
        windows = cfg.get("windows", [5, 20, 60])
    ann = cfg.get("annualization", 252)

    result = {}
    for w in windows:
        rv = returns.rolling(w, min_periods=max(3, w // 3)).std() * np.sqrt(ann)
        result[f"realized_vol_{w}d"] = rv

    return result


# ── Volatility Regime Classification ────────────────────────────────────────


def classify_vol_regime(
    vol_series: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    Classify volatility into regime: normal / elevated / panic.

    Uses rolling percentile thresholds.
    """
    cfg = _load_config().get("volatility", {}).get("regime_thresholds", {})
    normal_pct = cfg.get("normal_percentile", 0.60)
    elevated_pct = cfg.get("elevated_percentile", 0.80)

    rolling_pct = vol_series.rolling(lookback, min_periods=60).rank(pct=True)

    conditions = [
        rolling_pct >= elevated_pct,
        rolling_pct >= normal_pct,
    ]
    choices = ["panic", "elevated"]
    return pd.Series(
        np.select(conditions, choices, default="normal"),
        index=vol_series.index,
    )


# ── Full Volatility State ───────────────────────────────────────────────────


def build_volatility_state(
    inr_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build complete volatility state for all tickers.

    Returns long-format: date | ticker | ewma_vol | realized_vol |
                         vol_regime | vol_percentile
    """
    logger.info("Building volatility state...")

    # Pivot to wide returns
    prices_wide = inr_prices.pivot_table(
        index="date", columns="ticker", values="inr_price", aggfunc="first"
    )
    returns = prices_wide.pct_change().dropna(how="all")

    cfg = _load_config().get("volatility", {}).get("ewma", {})
    default_span = cfg.get("default_span", 60)

    # EWMA vol (primary)
    ewma = compute_ewma_volatility(returns, default_span)

    # Realized vol (20d)
    realized = returns.rolling(20, min_periods=5).std() * np.sqrt(252)

    frames = []
    for ticker in returns.columns:
        if ticker not in ewma.columns:
            continue
        df = pd.DataFrame({
            "date": returns.index,
            "ticker": ticker,
            "ewma_vol": ewma[ticker].values,
            "realized_vol": realized[ticker].values if ticker in realized.columns else np.nan,
        })

        # Vol regime
        df["vol_regime"] = classify_vol_regime(df["ewma_vol"]).values

        # Vol percentile (rolling 252d)
        df["vol_percentile"] = df["ewma_vol"].rolling(252, min_periods=60).rank(pct=True)

        frames.append(df.dropna(subset=["ewma_vol"]))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])

    n_tickers = result["ticker"].nunique()
    logger.info(f"  Volatility state: {n_tickers} tickers, {len(result)} rows")

    return result
