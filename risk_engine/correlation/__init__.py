"""
Risk Engine — Correlation Stress Engine.

Detects diversification collapse:
  - Rolling average pairwise correlation
  - Crisis clustering (all assets moving together)
  - Correlation expansion detection
  - Regime-conditional correlation snapshots
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


# ── Rolling Correlation ─────────────────────────────────────────────────────


def compute_rolling_avg_correlation(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Rolling average pairwise correlation across universe.

    Returns: date | avg_correlation | max_correlation | min_correlation |
             n_high_pairs
    """
    cfg = _load_config().get("correlation", {})
    spike_threshold = cfg.get("spike_threshold", 0.75)

    records = []
    for i in range(window, len(returns), 1):
        chunk = returns.iloc[i - window:i]
        corr = chunk.corr()

        # Upper triangle only (exclude diagonal)
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.values[mask]
        upper = upper[~np.isnan(upper)]

        if len(upper) == 0:
            continue

        n_high = int((np.abs(upper) > spike_threshold).sum())

        records.append({
            "date": returns.index[i],
            "avg_correlation": float(np.mean(upper)),
            "max_correlation": float(np.max(upper)),
            "min_correlation": float(np.min(upper)),
            "n_high_pairs": n_high,
            "pct_high_pairs": n_high / max(len(upper), 1),
        })

    return pd.DataFrame(records)


# ── Crisis Clustering Detection ─────────────────────────────────────────────


def detect_crisis_clustering(
    returns: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Detect periods where all assets move together (crisis clustering).

    High average correlation + negative returns = crisis clustering.

    Returns: date | is_clustering | avg_corr | avg_return | severity
    """
    records = []
    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window:i]
        corr = chunk.corr()

        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.values[mask]
        upper = upper[~np.isnan(upper)]

        avg_corr = float(np.mean(upper)) if len(upper) > 0 else 0
        avg_ret = float(chunk.mean().mean())
        is_clustering = avg_corr > threshold and avg_ret < 0

        severity = 0
        if is_clustering:
            severity = min(abs(avg_ret) * 100 * avg_corr, 1.0)

        records.append({
            "date": returns.index[i],
            "is_clustering": is_clustering,
            "avg_corr": avg_corr,
            "avg_return": avg_ret,
            "severity": severity,
        })

    return pd.DataFrame(records)


# ── Correlation Snapshots ───────────────────────────────────────────────────


def compute_correlation_snapshot(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Current correlation matrix from recent data."""
    return returns.iloc[-window:].corr()


def compute_regime_correlations(
    returns: pd.DataFrame,
    regime_states: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Compute correlation matrix conditional on each regime.

    Returns dict: regime_name → correlation_matrix
    """
    rs = regime_states.copy()
    rs["date"] = pd.to_datetime(rs["date"])
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    result = {}
    for regime in rs["regime"].unique():
        dates = rs[rs["regime"] == regime]["date"]
        regime_returns = returns[returns.index.isin(dates)]
        if len(regime_returns) >= 20:
            result[regime] = regime_returns.corr()

    return result


# ── Diversification Ratio ───────────────────────────────────────────────────


def compute_diversification_ratio(
    weights: np.ndarray | pd.Series,
    cov: pd.DataFrame,
) -> float:
    """
    Diversification ratio = weighted avg vol / portfolio vol.

    DR > 1 means diversification is working.
    DR ≈ 1 means no diversification benefit.
    """
    if isinstance(weights, pd.Series):
        weights = weights.values

    vols = np.sqrt(np.diag(cov.values))
    weighted_avg_vol = weights @ vols
    port_vol = np.sqrt(weights @ cov.values @ weights)

    if port_vol < 1e-10:
        return 1.0

    return float(weighted_avg_vol / port_vol)


def compute_rolling_diversification(
    returns: pd.DataFrame,
    weights: pd.Series | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Track diversification ratio over time.

    Returns: date | diversification_ratio
    """
    n = len(returns.columns)
    if weights is None:
        w = np.ones(n) / n
    else:
        w = weights.values

    records = []
    for i in range(window, len(returns), 5):
        cov = returns.iloc[i - window:i].cov()
        dr = compute_diversification_ratio(w, cov)
        records.append({"date": returns.index[i], "diversification_ratio": dr})

    return pd.DataFrame(records)
