"""
Risk Engine — Risk-Constrained Portfolio Construction.

The upgrade path:
  Sprint 1: HRP
  Sprint 2: HRP × regime
  Sprint 3: HRP × alpha × regime
  Sprint 4: HRP × alpha × regime × risk_budget × vol_scaling

This module integrates all four engines into the final allocation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omegaconf import OmegaConf

from risk_engine.budgeting import (
    adjust_weights_for_risk_budget,
    build_risk_budget_report,
)
from risk_engine.scaling import apply_vol_scaling, compute_vol_scaling_factor
from risk_engine.covariance import compute_regime_covariance, diagnose_covariance
from risk_engine.tail_risk import compute_cvar


def _load_config() -> dict:
    cfg_path = Path("configs/risk_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


def build_risk_aware_portfolio(
    base_weights: pd.DataFrame,
    returns: pd.DataFrame,
    alpha_scores: pd.DataFrame | None = None,
    regime_behavior=None,
    asset_types: pd.Series | None = None,
    max_tilt_strength: float = 0.30,
    min_confidence_to_tilt: float = 0.40,
) -> dict:
    """
    Full risk-aware portfolio construction pipeline.

    Pipeline:
      1. Start from HRP base weights
      2. Apply alpha tilt (if ML scores available)
      3. Select regime-aware covariance
      4. Adjust for risk budget constraints
      5. Apply volatility scaling
      6. Run diagnostics

    Returns
    -------
    dict with:
      - weights: final target weights (pd.Series)
      - allocation_df: full allocation DataFrame
      - risk_budget: risk contribution report
      - vol_scaling: scaling factor and details
      - covariance_diagnostics: condition number, etc.
      - cvar: portfolio CVaR
    """
    cfg = _load_config()
    logger.info("Building risk-aware portfolio...")

    # ── Step 1: Base weights ────────────────────────────────────────────
    if isinstance(base_weights, pd.DataFrame):
        weights = pd.Series(
            base_weights["target_weight"].values,
            index=base_weights["ticker"].values,
        )
    else:
        weights = base_weights.copy()

    logger.info(f"  Step 1: {len(weights)} base assets from HRP")

    # ── Step 2: Alpha tilt ──────────────────────────────────────────────
    if alpha_scores is not None and not alpha_scores.empty:
        from optimization.allocator import build_alpha_tilted_portfolio
        tilted_df = build_alpha_tilted_portfolio(
            base_weights if isinstance(base_weights, pd.DataFrame) else pd.DataFrame({
                "ticker": weights.index, "target_weight": weights.values
            }),
            alpha_scores,
            regime_behavior=regime_behavior,
            max_tilt_strength=max_tilt_strength,
            min_confidence_to_tilt=min_confidence_to_tilt,
        )
        weights = pd.Series(
            tilted_df["target_weight"].values,
            index=tilted_df["ticker"].values,
        )
        logger.info("  Step 2: Alpha tilt applied")
    else:
        logger.info("  Step 2: No alpha scores — skipping tilt")

    # ── Step 3: Regime-aware covariance ─────────────────────────────────
    regime = "risk_on"
    if regime_behavior is not None:
        regime = regime_behavior.regime

    common = weights.index.intersection(returns.columns)
    if common.empty:
        logger.warning("  No common tickers between weights and returns")
        return {"weights": weights, "status": "no_common_assets"}

    weights = weights[common]
    weights = weights / weights.sum()
    returns_aligned = returns[common]

    cov = compute_regime_covariance(returns_aligned, regime=regime)
    cov_diag = diagnose_covariance(cov)
    logger.info(f"  Step 3: Covariance ({regime}), condition={cov_diag['condition_number']:.0f}")

    # ── Step 4: Risk budget adjustment ──────────────────────────────────
    max_risk = cfg.get("constraints", {}).get("max_asset_risk_contribution", 0.35)
    weights = adjust_weights_for_risk_budget(weights, cov, max_risk)
    risk_report = build_risk_budget_report(weights, cov, asset_types)
    logger.info(f"  Step 4: Risk budget applied (max_risk_contrib={max_risk:.0%})")

    # ── Step 5: Volatility scaling ──────────────────────────────────────
    vol_cfg = cfg.get("volatility_targeting", {})
    if vol_cfg.get("enabled", True):
        port_vol = risk_report["portfolio_vol_annualized"]
        weights, scaling_factor = apply_vol_scaling(weights, port_vol)
        logger.info(f"  Step 5: Vol scaling={scaling_factor:.2f}, port_vol={port_vol:.2%}")
    else:
        scaling_factor = 1.0
        logger.info("  Step 5: Vol scaling disabled")

    # ── Step 6: CVaR check ──────────────────────────────────────────────
    port_returns = (returns_aligned * weights).sum(axis=1)
    portfolio_cvar = compute_cvar(port_returns, confidence=0.95)

    # ── Build allocation DataFrame ──────────────────────────────────────
    risk_pct = risk_report.get("risk_contributions", {})
    alloc_df = pd.DataFrame({
        "ticker": weights.index,
        "target_weight": weights.values,
        "risk_contribution": [risk_pct.get(t, 0) for t in weights.index],
        "strategy": "risk_aware_alpha_hrp",
    })

    cash = max(0, 1.0 - weights.sum())
    if cash > 0.01:
        cash_row = pd.DataFrame([{
            "ticker": "CASH",
            "target_weight": cash,
            "risk_contribution": 0.0,
            "strategy": "risk_aware_alpha_hrp",
        }])
        alloc_df = pd.concat([alloc_df, cash_row], ignore_index=True)

    result = {
        "weights": weights,
        "allocation_df": alloc_df,
        "risk_budget": risk_report,
        "vol_scaling": {
            "factor": scaling_factor,
            "portfolio_vol": risk_report["portfolio_vol_annualized"],
            "target_vol": vol_cfg.get("target_portfolio_vol", 0.12),
            "cash_allocation": cash,
        },
        "covariance_diagnostics": cov_diag,
        "portfolio_cvar": portfolio_cvar,
        "regime": regime,
        "status": "success",
    }

    logger.info(
        f"  Risk-aware portfolio complete: "
        f"vol={risk_report['portfolio_vol_annualized']:.2%}, "
        f"CVaR={portfolio_cvar:.2%}, "
        f"cash={cash:.1%}"
    )

    return result
