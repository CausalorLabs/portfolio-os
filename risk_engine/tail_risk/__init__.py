"""
Risk Engine — Tail Risk Engine.

Institutional portfolios don't blow up from average risk —
they blow up from tail risk.

Measures:
  - CVaR (Expected Shortfall) — what to expect when things go wrong
  - Semivariance — downside-only volatility
  - Tail Beta — how much worse you do vs market in crashes
  - Drawdown metrics — depth, duration, recovery
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


# ── CVaR (Expected Shortfall) ───────────────────────────────────────────────


def compute_cvar(
    returns: pd.Series | pd.DataFrame,
    confidence: float = 0.95,
) -> float | pd.Series:
    """
    CVaR (Conditional Value at Risk) = Expected Shortfall.

    "If we're in the worst 5% of days, how bad is it on average?"

    Parameters
    ----------
    returns : daily returns (Series for single asset, DataFrame for many)
    confidence : confidence level (0.95 = worst 5% tail)

    Returns
    -------
    CVaR as a negative number (more negative = worse)
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: compute_cvar(col, confidence))

    clean = returns.dropna()
    if len(clean) < 20:
        return np.nan

    var_threshold = clean.quantile(1 - confidence)
    tail = clean[clean <= var_threshold]

    return float(tail.mean()) if len(tail) > 0 else float(var_threshold)


def compute_var(
    returns: pd.Series | pd.DataFrame,
    confidence: float = 0.95,
) -> float | pd.Series:
    """Value at Risk at given confidence level."""
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: compute_var(col, confidence))

    clean = returns.dropna()
    if len(clean) < 20:
        return np.nan

    return float(clean.quantile(1 - confidence))


def compute_rolling_cvar(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.Series:
    """Rolling CVaR over time."""
    result = returns.rolling(window, min_periods=60).apply(
        lambda x: compute_cvar(pd.Series(x), confidence),
        raw=False,
    )
    return result


# ── Semivariance (Downside Risk) ────────────────────────────────────────────


def compute_semivariance(
    returns: pd.Series | pd.DataFrame,
    threshold: float = 0,
    annualize: bool = True,
) -> float | pd.Series:
    """
    Semivariance — variance of returns below threshold.

    Captures downside risk only. Standard deviation treats
    upside and downside equally (which is nonsense).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: compute_semivariance(col, threshold, annualize))

    clean = returns.dropna()
    downside = clean[clean < threshold]

    if len(downside) < 5:
        return 0.0

    semi_var = float(((downside - threshold) ** 2).mean())

    if annualize:
        semi_var *= 252

    return semi_var


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> float:
    """Sortino ratio = excess return / downside deviation."""
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    semi_var = compute_semivariance(returns, threshold=0, annualize=True)
    downside_dev = np.sqrt(semi_var)

    if downside_dev < 1e-10:
        return 0.0

    ann_return = excess.mean() * 252
    return float(ann_return / downside_dev)


# ── Tail Beta ───────────────────────────────────────────────────────────────


def compute_tail_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    tail_percentile: float = 0.10,
) -> float:
    """
    Tail beta — beta conditional on market being in left tail.

    "When the market crashes, how much worse does this asset do?"

    Higher tail_beta = more crash exposure.
    """
    aligned = pd.DataFrame({
        "asset": asset_returns,
        "market": market_returns,
    }).dropna()

    if len(aligned) < 30:
        return np.nan

    threshold = aligned["market"].quantile(tail_percentile)
    tail_days = aligned[aligned["market"] <= threshold]

    if len(tail_days) < 5:
        return np.nan

    market_tail_var = tail_days["market"].var()
    if market_tail_var < 1e-12:
        return np.nan

    cov_tail = tail_days[["asset", "market"]].cov().iloc[0, 1]
    return float(cov_tail / market_tail_var)


def compute_all_tail_betas(
    returns: pd.DataFrame,
    market_col: str | None = None,
    tail_percentile: float = 0.10,
) -> pd.Series:
    """
    Compute tail beta for all assets vs market proxy.

    If market_col is None, uses equal-weight portfolio as market.
    """
    if market_col and market_col in returns.columns:
        market = returns[market_col]
    else:
        market = returns.mean(axis=1)

    result = {}
    for ticker in returns.columns:
        result[ticker] = compute_tail_beta(returns[ticker], market, tail_percentile)

    return pd.Series(result, name="tail_beta")


# ── Drawdown Analysis ───────────────────────────────────────────────────────


def compute_drawdown_series(nav: pd.Series) -> pd.Series:
    """Rolling drawdown from peak."""
    peak = nav.cummax()
    return (nav - peak) / peak


def compute_max_drawdown(nav: pd.Series) -> float:
    """Maximum drawdown magnitude."""
    dd = compute_drawdown_series(nav)
    return float(dd.min())


def compute_drawdown_duration(nav: pd.Series) -> int:
    """Maximum drawdown duration in calendar days."""
    dd = compute_drawdown_series(nav)
    in_dd = dd < 0

    if not in_dd.any():
        return 0

    max_duration = 0
    current_start = None

    for i, (dt, is_dd) in enumerate(in_dd.items()):
        if is_dd and current_start is None:
            current_start = dt
        elif not is_dd and current_start is not None:
            duration = (dt - current_start).days
            max_duration = max(max_duration, duration)
            current_start = None

    # Still in drawdown
    if current_start is not None:
        duration = (nav.index[-1] - current_start).days
        max_duration = max(max_duration, duration)

    return max_duration


# ── Full Tail Risk Report ───────────────────────────────────────────────────


def build_tail_risk_report(
    returns: pd.DataFrame,
    nav: pd.Series | None = None,
) -> dict:
    """
    Comprehensive tail risk snapshot.

    Returns dict with CVaR, semivariance, tail betas, drawdown metrics.
    """
    cfg = _load_config().get("tail_risk", {})
    confidence = cfg.get("cvar_confidence", 0.95)

    report = {
        "cvar_per_asset": compute_cvar(returns, confidence).to_dict(),
        "var_per_asset": compute_var(returns, confidence).to_dict(),
        "semivariance_per_asset": compute_semivariance(returns).to_dict(),
        "tail_beta_per_asset": compute_all_tail_betas(returns).to_dict(),
    }

    # Portfolio-level (equal-weight proxy)
    port_returns = returns.mean(axis=1)
    report["portfolio_cvar"] = compute_cvar(port_returns, confidence)
    report["portfolio_sortino"] = compute_sortino_ratio(port_returns)

    if nav is not None:
        report["max_drawdown"] = compute_max_drawdown(nav)
        report["max_drawdown_duration_days"] = compute_drawdown_duration(nav)

    return report
