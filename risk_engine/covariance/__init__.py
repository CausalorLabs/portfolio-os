"""
Risk Engine — Dynamic Covariance Engine.

The heart of institutional risk modeling.

Estimates how assets move together RIGHT NOW,
not historically averaged forever.

Methods:
  1. EWMA Covariance — regime-adaptive, crisis-sensitive
  2. Ledoit-Wolf Shrinkage — stable, robust to noise
  3. Regime-Aware Selection — auto-selects method based on regime

During crises, correlations spike and diversification collapses.
This engine adapts to that reality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.covariance import LedoitWolf

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/risk_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ── EWMA Covariance ─────────────────────────────────────────────────────────


def compute_ewma_covariance(
    returns: pd.DataFrame,
    span: int = 60,
) -> pd.DataFrame:
    """
    EWMA covariance matrix — adapts faster to regime changes.

    Benefits: crisis sensitivity, smoother updates, faster response.
    """
    ewm = returns.ewm(span=span, min_periods=max(10, span // 3))
    cov = ewm.cov().iloc[-len(returns.columns):]

    # Clean up multi-index
    if isinstance(cov.index, pd.MultiIndex):
        last_date = cov.index.get_level_values(0)[-1]
        cov = cov.loc[last_date]

    return cov


# ── Ledoit-Wolf Shrinkage ───────────────────────────────────────────────────


def compute_shrinkage_covariance(
    returns: pd.DataFrame,
    window: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Ledoit-Wolf shrinkage covariance — dramatically more stable.

    Sample covariance matrices are extremely unstable, especially with
    many assets, limited observations, or stress periods.
    Shrinkage corrects this.

    Returns
    -------
    (covariance_matrix, shrinkage_coefficient)
    """
    if window is not None:
        data = returns.iloc[-window:]
    else:
        data = returns

    data_clean = data.dropna(axis=0, how="any")

    if len(data_clean) < 10:
        logger.warning("Insufficient data for shrinkage — falling back to sample")
        cov = data_clean.cov()
        return cov, 0.0

    lw = LedoitWolf()
    lw.fit(data_clean.values)

    cov = pd.DataFrame(
        lw.covariance_,
        index=data_clean.columns,
        columns=data_clean.columns,
    )

    logger.debug(f"  Shrinkage coefficient: {lw.shrinkage_:.4f}")
    return cov, float(lw.shrinkage_)


# ── Sample Covariance ───────────────────────────────────────────────────────


def compute_sample_covariance(
    returns: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Raw sample covariance (baseline, unstable)."""
    if window is not None:
        data = returns.iloc[-window:]
    else:
        data = returns
    return data.cov()


# ── Regime-Aware Covariance Selection ───────────────────────────────────────


def compute_regime_covariance(
    returns: pd.DataFrame,
    regime: str = "risk_on",
    ewma_span: int | None = None,
) -> pd.DataFrame:
    """
    Auto-select covariance method based on current market regime.

    Panic/high_vol → EWMA (fast-adapting)
    Risk_on/risk_off → Shrinkage (stable)
    """
    cfg = _load_config().get("covariance", {})
    override = cfg.get("regime_override", {})
    method = override.get(regime, cfg.get("default_method", "shrinkage"))

    if ewma_span is None:
        ewma_span = cfg.get("ewma_span", 60)

    if method == "ewma":
        cov = compute_ewma_covariance(returns, span=ewma_span)
        logger.info(f"  Covariance: EWMA (span={ewma_span}) for regime={regime}")
    elif method == "shrinkage":
        cov, coeff = compute_shrinkage_covariance(returns)
        logger.info(f"  Covariance: Ledoit-Wolf (shrinkage={coeff:.4f}) for regime={regime}")
    else:
        cov = compute_sample_covariance(returns)
        logger.info(f"  Covariance: sample for regime={regime}")

    return cov


# ── Covariance Diagnostics ──────────────────────────────────────────────────


def diagnose_covariance(cov: pd.DataFrame) -> dict:
    """
    Diagnostic checks on a covariance matrix.

    Returns: condition_number, is_positive_definite, max_correlation,
             avg_correlation, determinant
    """
    eigenvalues = np.linalg.eigvalsh(cov.values)

    # Correlation from covariance
    std = np.sqrt(np.diag(cov.values))
    std_outer = np.outer(std, std)
    corr = cov.values / np.where(std_outer > 0, std_outer, 1)
    np.fill_diagonal(corr, 0)

    return {
        "condition_number": float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)),
        "is_positive_definite": bool(eigenvalues.min() > 0),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "max_correlation": float(np.abs(corr).max()),
        "avg_correlation": float(np.abs(corr).mean()),
        "determinant": float(np.linalg.det(cov.values)),
        "n_assets": len(cov),
    }


# ── Rolling Covariance ──────────────────────────────────────────────────────


def compute_rolling_covariance_stats(
    returns: pd.DataFrame,
    window: int = 60,
    method: str = "ewma",
) -> pd.DataFrame:
    """
    Track covariance stability over time.

    Returns: date | avg_correlation | max_correlation | condition_number |
             portfolio_vol
    """
    dates = returns.index[window:]
    records = []

    for i in range(window, len(returns), 5):  # every 5 days for speed
        chunk = returns.iloc[i - window:i]
        dt = returns.index[i]

        if method == "ewma":
            cov = compute_ewma_covariance(chunk, span=window)
        else:
            cov = chunk.cov()

        diag = diagnose_covariance(cov)

        # Equal-weight portfolio vol
        n = len(cov)
        w = np.ones(n) / n
        port_var = w @ cov.values @ w
        port_vol = np.sqrt(port_var) * np.sqrt(252)

        records.append({
            "date": dt,
            "avg_correlation": diag["avg_correlation"],
            "max_correlation": diag["max_correlation"],
            "condition_number": diag["condition_number"],
            "portfolio_vol": port_vol,
        })

    return pd.DataFrame(records)
