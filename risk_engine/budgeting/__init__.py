"""
Risk Engine — Risk Budgeting Framework.

Allocate risk, not just capital.

A portfolio with 60% equities might have 90%+ of its risk
from equities. Risk budgeting makes that explicit.

Measures:
  - Marginal Risk Contribution (MRC) — risk added by marginal unit
  - Total Risk Contribution (TRC) — total risk from each asset
  - Risk parity verification — are risk contributions balanced?
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


# ── Risk Contribution ───────────────────────────────────────────────────────


def compute_marginal_risk_contribution(
    weights: np.ndarray | pd.Series,
    cov: pd.DataFrame,
) -> pd.Series:
    """
    Marginal Risk Contribution = (cov @ w) / portfolio_vol.

    How much one additional unit of this asset changes portfolio risk.
    """
    if isinstance(weights, pd.Series):
        tickers = weights.index
        w = weights.values
    else:
        tickers = cov.columns
        w = weights

    sigma = cov.values
    port_var = w @ sigma @ w
    port_vol = np.sqrt(port_var)

    if port_vol < 1e-12:
        return pd.Series(np.zeros(len(w)), index=tickers, name="mrc")

    mrc = (sigma @ w) / port_vol
    return pd.Series(mrc, index=tickers, name="mrc")


def compute_total_risk_contribution(
    weights: np.ndarray | pd.Series,
    cov: pd.DataFrame,
) -> pd.Series:
    """
    Total Risk Contribution = weight × MRC.

    What fraction of portfolio risk comes from each asset.
    Sum(TRC) = portfolio_vol.
    """
    if isinstance(weights, pd.Series):
        tickers = weights.index
        w = weights.values
    else:
        tickers = cov.columns
        w = weights

    mrc = compute_marginal_risk_contribution(w, cov)
    trc = w * mrc.values

    return pd.Series(trc, index=tickers, name="trc")


def compute_risk_contribution_pct(
    weights: np.ndarray | pd.Series,
    cov: pd.DataFrame,
) -> pd.Series:
    """
    Percentage risk contribution — TRC as % of total portfolio vol.

    This is the key risk budgeting number.
    """
    trc = compute_total_risk_contribution(weights, cov)
    total = trc.sum()

    if abs(total) < 1e-12:
        return pd.Series(np.zeros(len(trc)), index=trc.index, name="risk_pct")

    return pd.Series(trc / total, index=trc.index, name="risk_pct")


# ── Risk Parity Check ───────────────────────────────────────────────────────


def check_risk_parity(
    weights: np.ndarray | pd.Series,
    cov: pd.DataFrame,
) -> dict:
    """
    Evaluate how close a portfolio is to risk parity.

    Returns:
      - risk_contributions: per-asset risk %
      - max_risk_asset: highest risk contributor
      - max_risk_pct: its share
      - is_balanced: whether max < threshold
      - herfindahl_risk: risk concentration (0=equal, 1=concentrated)
    """
    cfg = _load_config().get("risk_budgeting", {})
    max_contrib = cfg.get("max_risk_contribution", 0.40)

    risk_pct = compute_risk_contribution_pct(weights, cov)

    # Herfindahl index of risk contributions
    herfindahl = float((risk_pct ** 2).sum())
    n = len(risk_pct)
    normalized_hhi = (herfindahl - 1 / n) / (1 - 1 / n) if n > 1 else 0

    max_asset = risk_pct.idxmax()
    max_pct = float(risk_pct.max())

    return {
        "risk_contributions": risk_pct.to_dict(),
        "max_risk_asset": max_asset,
        "max_risk_pct": max_pct,
        "is_balanced": max_pct <= max_contrib,
        "herfindahl_risk": normalized_hhi,
        "n_assets": n,
    }


# ── Risk-Budget-Aware Weight Adjustment ─────────────────────────────────────


def adjust_weights_for_risk_budget(
    weights: pd.Series,
    cov: pd.DataFrame,
    max_risk_contribution: float | None = None,
    iterations: int = 10,
) -> pd.Series:
    """
    Adjust weights to respect risk budget constraints.

    Iteratively reduces weight of assets with excessive risk contribution
    and redistributes to under-contributing assets.
    """
    cfg = _load_config().get("risk_budgeting", {})
    if max_risk_contribution is None:
        max_risk_contribution = cfg.get("max_risk_contribution", 0.40)

    w = weights.copy()
    common = w.index.intersection(cov.columns)
    w = w[common]
    cov_aligned = cov.loc[common, common]

    for _ in range(iterations):
        risk_pct = compute_risk_contribution_pct(w, cov_aligned)
        breaches = risk_pct[risk_pct > max_risk_contribution]

        if breaches.empty:
            break

        # Reduce over-contributing assets
        for ticker, contrib in breaches.items():
            excess = contrib - max_risk_contribution
            reduction = excess * 0.5  # dampen to avoid oscillation
            w[ticker] *= (1 - reduction)

        # Re-normalize
        w = w / w.sum()

    logger.debug(f"  Risk budget adjustment: {len(weights)} → {len(w)} assets")
    return w


# ── Risk Decomposition Report ───────────────────────────────────────────────


def build_risk_budget_report(
    weights: pd.Series,
    cov: pd.DataFrame,
    asset_types: pd.Series | None = None,
) -> dict:
    """
    Full risk budget report.

    Returns: per-asset risk %, sector risk %, parity check, portfolio vol.
    """
    common = weights.index.intersection(cov.columns)
    w = weights[common]
    cov_aligned = cov.loc[common, common]

    risk_pct = compute_risk_contribution_pct(w, cov_aligned)
    parity = check_risk_parity(w, cov_aligned)

    # Portfolio vol
    port_vol = float(np.sqrt(w.values @ cov_aligned.values @ w.values) * np.sqrt(252))

    report = {
        "portfolio_vol_annualized": port_vol,
        "risk_contributions": risk_pct.to_dict(),
        "parity_check": parity,
    }

    # Sector-level risk contributions
    if asset_types is not None:
        sector_risk = {}
        for sector in asset_types.unique():
            sector_tickers = asset_types[asset_types == sector].index
            in_common = sector_tickers.intersection(risk_pct.index)
            sector_risk[sector] = float(risk_pct[in_common].sum())
        report["sector_risk"] = sector_risk

    return report
