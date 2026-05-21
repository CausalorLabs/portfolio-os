"""
ML Alpha Engine — Extended Feature Store.

Expands the base feature store with ML-specific features:
  - Cross-sectional features (relative ranks, percentiles)
  - Regime-injected features (state, confidence, stress)
  - Factor features (beta, downside vol ratio)
  - Macro features (VIX, cross-asset correlation)

Outputs wide-format DataFrame ready for ML training.
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


# ── Cross-sectional features ────────────────────────────────────────────────


def compute_cross_sectional_features(feature_store: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional ranks and percentiles.

    For each date, rank each ticker's feature value relative to the universe.
    Returns long-format additions: date | ticker | feature | value
    """
    rank_features = {
        "volatility_20d": "relative_vol_rank",
        "rsi_14": "relative_rsi_rank",
        "trend_slope_60d": "relative_trend_rank",
    }

    frames = []
    for src_feat, rank_name in rank_features.items():
        sub = feature_store[feature_store["feature"] == src_feat].copy()
        if sub.empty:
            continue
        sub[rank_name] = sub.groupby("date")["value"].rank(pct=True)
        ranked = sub[["date", "ticker", rank_name]].melt(
            id_vars=["date", "ticker"],
            var_name="feature",
            value_name="value",
        )
        frames.append(ranked)

    if frames:
        result = pd.concat(frames, ignore_index=True)
        logger.info(f"  Cross-sectional features: {len(rank_features)} added")
        return result
    return pd.DataFrame(columns=["date", "ticker", "feature", "value"])


# ── Factor features ─────────────────────────────────────────────────────────


def compute_beta_features(inr_prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling beta and downside vol ratio per ticker.

    Beta = Cov(asset, market) / Var(market)
    Market proxy = equal-weighted portfolio of all tickers.
    """
    df = inr_prices[["date", "ticker", "inr_price"]].copy()
    df = df.sort_values(["ticker", "date"])

    # Compute daily returns
    df["ret"] = df.groupby("ticker")["inr_price"].pct_change()

    # Market return = equal-weighted average
    mkt = df.groupby("date")["ret"].mean().rename("mkt_ret").reset_index()
    df = df.merge(mkt, on="date", how="left")

    frames = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").copy()

        # Rolling beta
        cov = grp["ret"].rolling(window, min_periods=30).cov(grp["mkt_ret"])
        var_mkt = grp["mkt_ret"].rolling(window, min_periods=30).var()
        grp["beta_60d"] = cov / var_mkt.clip(lower=1e-10)

        # Downside vol ratio: downside_vol / total_vol
        grp["downside_vol"] = grp["ret"].clip(upper=0).rolling(window, min_periods=30).std()
        grp["total_vol"] = grp["ret"].rolling(window, min_periods=30).std()
        grp["downside_vol_ratio"] = grp["downside_vol"] / grp["total_vol"].clip(lower=1e-10)

        for col in ["beta_60d", "downside_vol_ratio"]:
            sub = grp[["date"]].copy()
            sub["ticker"] = ticker
            sub["feature"] = col
            sub["value"] = grp[col].values
            frames.append(sub.dropna(subset=["value"]))

    if frames:
        result = pd.concat(frames, ignore_index=True)
        logger.info(f"  Factor features: beta_60d, downside_vol_ratio ({len(result)} rows)")
        return result
    return pd.DataFrame(columns=["date", "ticker", "feature", "value"])


# ── Regime-injected features ────────────────────────────────────────────────


def compute_regime_features_for_ml(regime_states: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Convert regime pipeline outputs into ML features.

    Features: regime_state_encoded, regime_confidence, regime_vol_ratio,
              regime_breadth, regime_momentum
    """
    if regime_states is None or regime_states.empty:
        return pd.DataFrame(columns=["date", "feature", "value"])

    # Try loading regime features for richer data
    regime_feat_path = Path("data/processed/regime_features.parquet")
    if regime_feat_path.exists():
        rf = pd.read_parquet(regime_feat_path)
        rf["date"] = pd.to_datetime(rf["date"])
    else:
        rf = None

    rs = regime_states.copy()
    rs["date"] = pd.to_datetime(rs["date"])

    frames = []

    # Regime state as numeric
    regime_map = {"risk_on": 1, "risk_off": 0, "panic": -1, "high_vol": -0.5}
    encoded = rs[["date"]].copy()
    encoded["feature"] = "regime_state_encoded"
    encoded["value"] = rs["regime"].map(regime_map).fillna(0)
    frames.append(encoded)

    # Confidence
    if "confidence" in rs.columns:
        conf = rs[["date"]].copy()
        conf["feature"] = "regime_confidence"
        conf["value"] = rs["confidence"]
        frames.append(conf)

    # Rich regime features from the feature pipeline
    if rf is not None:
        for col in ["vol_regime_ratio", "breadth_score", "spy_momentum"]:
            if col in rf.columns:
                sub = rf[["date"]].copy()
                feat_name = {
                    "vol_regime_ratio": "regime_vol_ratio",
                    "breadth_score": "regime_breadth",
                    "spy_momentum": "regime_momentum",
                }.get(col, col)
                sub["feature"] = feat_name
                sub["value"] = rf[col].values
                frames.append(sub.dropna(subset=["value"]))

    if frames:
        result = pd.concat(frames, ignore_index=True)
        logger.info(f"  Regime ML features: {result['feature'].nunique()} types")
        return result
    return pd.DataFrame(columns=["date", "feature", "value"])


# ── Macro features ──────────────────────────────────────────────────────────


def compute_macro_features() -> pd.DataFrame:
    """
    Lightweight macro features from available regime data.

    Uses VIX, cross-asset correlation from the regime feature pipeline.
    """
    rf_path = Path("data/processed/regime_features.parquet")
    if not rf_path.exists():
        return pd.DataFrame(columns=["date", "feature", "value"])

    rf = pd.read_parquet(rf_path)
    rf["date"] = pd.to_datetime(rf["date"])

    frames = []
    for col, feat_name in [("vix", "vix_level"), ("vix_zscore", "vix_zscore"),
                            ("cross_asset_corr", "cross_asset_corr")]:
        if col in rf.columns:
            sub = rf[["date"]].copy()
            sub["feature"] = feat_name
            sub["value"] = rf[col].values
            frames.append(sub.dropna(subset=["value"]))

    if frames:
        result = pd.concat(frames, ignore_index=True)
        logger.info(f"  Macro features: {result['feature'].nunique()} types")
        return result
    return pd.DataFrame(columns=["date", "feature", "value"])


# ── Assembler ────────────────────────────────────────────────────────────────


def build_extended_feature_store(
    base_store: pd.DataFrame,
    inr_prices: pd.DataFrame,
    regime_states: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the extended feature store combining base + ML features.

    Parameters
    ----------
    base_store : Long-format feature store (date, ticker, feature, value)
    inr_prices : Raw price data (date, ticker, inr_price)
    regime_states : Optional regime data (date, regime, confidence)

    Returns
    -------
    Extended long-format feature store
    """
    logger.info("Building extended feature store for ML...")

    frames = [base_store]

    # Cross-sectional features
    cs = compute_cross_sectional_features(base_store)
    if not cs.empty:
        frames.append(cs)

    # Factor features (beta, downside vol)
    beta = compute_beta_features(inr_prices)
    if not beta.empty:
        frames.append(beta)

    # Regime features (broadcast to all tickers)
    regime_ml = compute_regime_features_for_ml(regime_states)
    if not regime_ml.empty:
        tickers = base_store["ticker"].unique()
        regime_expanded = []
        for ticker in tickers:
            t_df = regime_ml.copy()
            t_df["ticker"] = ticker
            regime_expanded.append(t_df)
        regime_all = pd.concat(regime_expanded, ignore_index=True)
        frames.append(regime_all)

    # Macro features (broadcast to all tickers)
    macro = compute_macro_features()
    if not macro.empty:
        tickers = base_store["ticker"].unique()
        macro_expanded = []
        for ticker in tickers:
            m_df = macro.copy()
            m_df["ticker"] = ticker
            macro_expanded.append(m_df)
        macro_all = pd.concat(macro_expanded, ignore_index=True)
        frames.append(macro_all)

    extended = pd.concat(frames, ignore_index=True)
    extended = extended.sort_values(["date", "ticker", "feature"]).reset_index(drop=True)

    n_features = extended["feature"].nunique()
    n_tickers = extended["ticker"].nunique()
    logger.info(f"  Extended store: {n_features} features × {n_tickers} tickers = {len(extended)} rows")

    return extended
