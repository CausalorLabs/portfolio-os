"""
Risk Engine — Pipeline Orchestrator.

Sprint 4: Dynamic Risk & Covariance Engine.

Replaces static risk assumptions with adaptive risk modeling:
  - Dynamic volatility (EWMA multi-horizon)
  - Regime-aware covariance (auto-selects method)
  - Correlation stress detection
  - Tail risk (CVaR, semivariance, tail beta)
  - Risk budgeting (marginal/total risk contribution)
  - Volatility targeting (dynamic position sizing)
  - Stress testing (historical + synthetic)
  - Risk-constrained portfolio construction
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from omegaconf import OmegaConf

from risk_engine.volatility import build_volatility_state
from risk_engine.covariance import compute_regime_covariance, diagnose_covariance
from risk_engine.correlation import (
    compute_rolling_avg_correlation,
    detect_crisis_clustering,
    compute_diversification_ratio,
)
from risk_engine.tail_risk import build_tail_risk_report
from risk_engine.budgeting import build_risk_budget_report
from risk_engine.scaling import build_scaling_report
from risk_engine.stress_testing import build_stress_test_report
from risk_engine.constraints import build_risk_aware_portfolio


def _load_config() -> dict:
    cfg_path = Path("configs/risk_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


def run_risk_pipeline(
    inr_prices: pd.DataFrame,
    base_weights: pd.DataFrame,
    alpha_scores: pd.DataFrame | None = None,
    regime_behavior=None,
    asset_types: pd.Series | None = None,
    nav: pd.Series | None = None,
) -> dict:
    """
    Full risk engine pipeline.

    Steps:
      1. Build volatility state (EWMA, realized, regime)
      2. Compute regime-aware covariance
      3. Detect correlation stress
      4. Compute tail risk
      5. Risk budgeting
      6. Volatility targeting
      7. Build risk-constrained portfolio
      8. Stress testing

    Parameters
    ----------
    inr_prices : Long-format: date | ticker | inr_price
    base_weights : DataFrame with ticker, target_weight
    alpha_scores : Optional ML alpha scores
    regime_behavior : Optional RegimeBehavior from regime engine
    asset_types : Optional Series mapping ticker → asset_type
    nav : Optional portfolio NAV series

    Returns
    -------
    dict with all risk engine outputs
    """
    logger.info("=" * 60)
    logger.info("RISK ENGINE — Dynamic Risk & Covariance Pipeline")
    logger.info("=" * 60)

    results = {}

    # ── Step 1: Volatility State ────────────────────────────────────────
    logger.info("[1/8] Building volatility state...")
    vol_state = build_volatility_state(inr_prices)
    results["volatility_state"] = vol_state
    logger.info(f"  → {len(vol_state)} volatility observations")

    # ── Prepare wide returns ────────────────────────────────────────────
    prices_wide = inr_prices.pivot_table(
        index="date", columns="ticker", values="inr_price", aggfunc="first"
    )
    returns = prices_wide.pct_change().dropna(how="all")

    # ── Step 2: Regime-Aware Covariance ─────────────────────────────────
    logger.info("[2/8] Computing regime-aware covariance...")
    regime = "risk_on"
    if regime_behavior is not None:
        regime = regime_behavior.regime

    cov = compute_regime_covariance(returns, regime=regime)
    cov_diag = diagnose_covariance(cov)
    results["covariance"] = cov
    results["covariance_diagnostics"] = cov_diag
    logger.info(f"  → Condition number: {cov_diag['condition_number']:.0f}")

    # ── Step 3: Correlation Stress ──────────────────────────────────────
    logger.info("[3/8] Detecting correlation stress...")
    corr_rolling = compute_rolling_avg_correlation(returns, window=60)
    clustering = detect_crisis_clustering(returns, window=20)
    results["correlation_rolling"] = corr_rolling
    results["crisis_clustering"] = clustering

    if not clustering.empty:
        recent_cluster = clustering.iloc[-1]
        is_clustering = recent_cluster.get("is_clustering", False)
        logger.info(f"  → Crisis clustering active: {is_clustering}")

    # ── Step 4: Tail Risk ───────────────────────────────────────────────
    logger.info("[4/8] Computing tail risk metrics...")
    tail = build_tail_risk_report(returns, nav)
    results["tail_risk"] = tail
    logger.info(f"  → Portfolio CVaR: {tail.get('portfolio_cvar', 'N/A')}")

    # ── Step 5: Risk-Constrained Portfolio ──────────────────────────────
    logger.info("[5/8] Building risk-constrained portfolio...")
    portfolio = build_risk_aware_portfolio(
        base_weights=base_weights,
        returns=returns,
        alpha_scores=alpha_scores,
        regime_behavior=regime_behavior,
        asset_types=asset_types,
    )
    results["portfolio"] = portfolio
    results["final_weights"] = portfolio.get("weights")
    results["allocation"] = portfolio.get("allocation_df")

    # ── Step 6: Risk Budgeting ──────────────────────────────────────────
    logger.info("[6/8] Computing risk budget...")
    if portfolio.get("weights") is not None and cov is not None:
        common = portfolio["weights"].index.intersection(cov.columns)
        if not common.empty:
            risk_budget = build_risk_budget_report(
                portfolio["weights"][common],
                cov.loc[common, common],
                asset_types,
            )
            results["risk_budget"] = risk_budget
            logger.info(f"  → Portfolio vol: {risk_budget['portfolio_vol_annualized']:.2%}")

    # ── Step 7: Vol Scaling Report ──────────────────────────────────────
    logger.info("[7/8] Volatility scaling report...")
    if "risk_budget" in results:
        port_vol = results["risk_budget"]["portfolio_vol_annualized"]
        scaling = build_scaling_report(port_vol)
        results["vol_scaling"] = scaling
        logger.info(f"  → Scaling factor: {scaling['scaling_factor']:.2f}")

    # ── Step 8: Stress Testing ──────────────────────────────────────────
    logger.info("[8/8] Running stress tests...")
    weights_series = portfolio.get("weights", pd.Series(dtype=float))
    stress = build_stress_test_report(
        weights=weights_series,
        returns=returns,
        cov=cov,
        asset_types=asset_types,
    )
    results["stress_tests"] = stress
    hist_count = len(stress.get("historical_scenarios", []))
    synth_count = len(stress.get("synthetic_scenarios", []))
    logger.info(f"  → {hist_count} historical + {synth_count} synthetic scenarios")

    logger.info("=" * 60)
    logger.info("RISK ENGINE COMPLETE")
    logger.info("=" * 60)

    return results
