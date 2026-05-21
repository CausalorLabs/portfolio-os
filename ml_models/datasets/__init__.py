"""
ML Alpha Engine — Dataset & Target Construction.

Builds supervised learning targets for cross-sectional ranking.
Three targets aligned with portfolio utility:

1. Forward Rank      — 5D relative performance across universe
2. Risk-Adjusted     — 20D forward return / forward volatility
3. Downside Prob     — P(negative return) over forward horizon

IMPORTANT: All targets use FUTURE data → only used for training labels.
Features must NEVER include forward-looking information.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/ml_alpha.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ── Target builders ──────────────────────────────────────────────────────────


def compute_forward_returns(
    inr_prices: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Compute forward returns for each ticker over a given horizon.

    Returns DataFrame: date | ticker | forward_return_{horizon}d
    """
    df = inr_prices[["date", "ticker", "inr_price"]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    col = f"forward_return_{horizon}d"
    df[col] = df.groupby("ticker")["inr_price"].transform(
        lambda x: x.shift(-horizon) / x - 1
    )
    return df[["date", "ticker", col]].dropna()


def compute_forward_rank(
    inr_prices: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Target 1 — Cross-sectional percentile rank of forward returns.

    Returns DataFrame: date | ticker | forward_rank_{horizon}d (0-1)
    """
    fwd = compute_forward_returns(inr_prices, horizon)
    ret_col = f"forward_return_{horizon}d"
    rank_col = f"forward_rank_{horizon}d"

    fwd[rank_col] = fwd.groupby("date")[ret_col].rank(pct=True)
    return fwd[["date", "ticker", rank_col]]


def compute_risk_adjusted_target(
    inr_prices: pd.DataFrame,
    horizon: int = 20,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Target 2 — Forward return / forward volatility.

    Measures quality of returns, not just magnitude.
    Returns DataFrame: date | ticker | risk_adjusted_{horizon}d
    """
    df = inr_prices[["date", "ticker", "inr_price"]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily returns
    df["daily_ret"] = df.groupby("ticker")["inr_price"].pct_change()

    # Forward return over horizon
    df["fwd_ret"] = df.groupby("ticker")["inr_price"].transform(
        lambda x: x.shift(-horizon) / x - 1
    )

    # Forward realized volatility (annualized)
    df["fwd_vol"] = df.groupby("ticker")["daily_ret"].transform(
        lambda x: x.shift(-1).rolling(vol_window).std().shift(-vol_window + 1) * np.sqrt(252)
    )

    col = f"risk_adjusted_{horizon}d"
    df[col] = df["fwd_ret"] / df["fwd_vol"].clip(lower=0.01)

    # Cross-sectional rank for consistency
    rank_col = f"risk_adjusted_rank_{horizon}d"
    df[rank_col] = df.groupby("date")[col].rank(pct=True)

    return df[["date", "ticker", col, rank_col]].dropna()


def compute_downside_probability(
    inr_prices: pd.DataFrame,
    horizon: int = 20,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Target 3 — Empirical probability of negative returns.

    Uses rolling lookback window of forward returns to estimate P(ret < 0).
    Returns DataFrame: date | ticker | downside_prob_{horizon}d
    """
    df = inr_prices[["date", "ticker", "inr_price"]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["fwd_ret"] = df.groupby("ticker")["inr_price"].transform(
        lambda x: x.shift(-horizon) / x - 1
    )

    col = f"downside_prob_{horizon}d"
    df[col] = df.groupby("ticker")["fwd_ret"].transform(
        lambda x: x.rolling(lookback, min_periods=60).apply(
            lambda w: (w < 0).mean(), raw=True
        )
    )

    return df[["date", "ticker", col]].dropna()


# ── Dataset assembly ─────────────────────────────────────────────────────────


def build_ml_dataset(
    inr_prices: pd.DataFrame,
    feature_store: pd.DataFrame,
    regime_states: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble the full ML dataset: features (X) + targets (y).

    Parameters
    ----------
    inr_prices : DataFrame with date, ticker, inr_price
    feature_store : Long-format features (date, ticker, feature, value)
    regime_states : Optional regime data (date, regime, confidence)

    Returns
    -------
    DataFrame in wide format: date | ticker | feature_1 | ... | target_1 | ...
    """
    cfg = _load_config()
    target_cfg = cfg.get("targets", {})

    logger.info("Building ML dataset...")

    # ── Pivot features to wide format ────────────────────────────────────
    features_wide = feature_store.pivot_table(
        index=["date", "ticker"],
        columns="feature",
        values="value",
        aggfunc="first",
    ).reset_index()
    features_wide.columns.name = None

    logger.info(f"  Features: {features_wide.shape[1] - 2} columns, {len(features_wide)} rows")

    # ── Compute targets ──────────────────────────────────────────────────
    rank_horizon = target_cfg.get("forward_rank", {}).get("horizon_days", 5)
    ra_horizon = target_cfg.get("risk_adjusted", {}).get("horizon_days", 20)
    ra_vol = target_cfg.get("risk_adjusted", {}).get("vol_window", 20)
    dp_horizon = target_cfg.get("downside_probability", {}).get("horizon_days", 20)

    targets = []

    rank_df = compute_forward_rank(inr_prices, rank_horizon)
    targets.append(rank_df)
    logger.info(f"  Target: forward_rank_{rank_horizon}d ({len(rank_df)} rows)")

    ra_df = compute_risk_adjusted_target(inr_prices, ra_horizon, ra_vol)
    targets.append(ra_df)
    logger.info(f"  Target: risk_adjusted_{ra_horizon}d ({len(ra_df)} rows)")

    dp_df = compute_downside_probability(inr_prices, dp_horizon)
    targets.append(dp_df)
    logger.info(f"  Target: downside_prob_{dp_horizon}d ({len(dp_df)} rows)")

    # ── Merge features + targets ─────────────────────────────────────────
    dataset = features_wide.copy()
    for t_df in targets:
        dataset = dataset.merge(t_df, on=["date", "ticker"], how="left")

    # ── Inject regime features ───────────────────────────────────────────
    if regime_states is not None and not regime_states.empty:
        rs = regime_states[["date", "regime", "confidence"]].copy()
        rs["date"] = pd.to_datetime(rs["date"])

        # Encode regime as numeric
        regime_map = {"risk_on": 1, "risk_off": 0, "panic": -1, "high_vol": -0.5}
        rs["regime_state_encoded"] = rs["regime"].map(regime_map).fillna(0)
        rs["regime_confidence"] = rs["confidence"]

        dataset = dataset.merge(
            rs[["date", "regime_state_encoded", "regime_confidence"]],
            on="date",
            how="left",
        )
        logger.info("  Injected regime features")

    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Drop rows where primary target is missing
    primary_target = f"forward_rank_{rank_horizon}d"
    before = len(dataset)
    dataset = dataset.dropna(subset=[primary_target])
    logger.info(f"  Final dataset: {len(dataset)} rows ({before - len(dataset)} dropped for missing target)")

    return dataset
