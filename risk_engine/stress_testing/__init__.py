"""
Risk Engine — Stress Testing Engine.

What happens to our portfolio if 2008 happens again?
What if India crashes 30% while USD spikes 15%?

Two modes:
  1. Historical Replay — apply actual crisis returns
  2. Synthetic Shocks — apply hypothetical scenarios
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


# ── Historical Stress Testing ───────────────────────────────────────────────


def apply_historical_scenario(
    weights: pd.Series,
    returns: pd.DataFrame,
    start: str,
    end: str,
    scenario_name: str = "historical",
) -> dict:
    """
    Apply a historical crisis period to current portfolio weights.

    Finds the returns during [start, end], computes portfolio impact.
    """
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    crisis = returns.loc[start:end]

    if crisis.empty:
        logger.warning(f"  No data for scenario {scenario_name} ({start} to {end})")
        return {
            "scenario": scenario_name,
            "portfolio_return": np.nan,
            "worst_day": np.nan,
            "status": "no_data",
        }

    common = weights.index.intersection(crisis.columns)
    if common.empty:
        return {
            "scenario": scenario_name,
            "portfolio_return": np.nan,
            "status": "no_common_assets",
        }

    w = weights[common] / weights[common].sum()  # re-normalize
    crisis_aligned = crisis[common]

    # Portfolio daily returns during crisis
    port_returns = (crisis_aligned * w).sum(axis=1)
    cum_return = float((1 + port_returns).prod() - 1)
    worst_day = float(port_returns.min())
    best_day = float(port_returns.max())
    vol_during = float(port_returns.std() * np.sqrt(252))

    # Per-asset impact
    asset_impacts = {}
    for ticker in common:
        asset_cum = float((1 + crisis_aligned[ticker]).prod() - 1)
        asset_impacts[ticker] = asset_cum

    return {
        "scenario": scenario_name,
        "portfolio_return": cum_return,
        "worst_day": worst_day,
        "best_day": best_day,
        "volatility_during": vol_during,
        "duration_days": len(crisis),
        "asset_impacts": asset_impacts,
        "status": "computed",
    }


def run_historical_stress_tests(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Run all configured historical scenarios."""
    cfg = _load_config().get("stress_testing", {})
    scenarios = cfg.get("historical_scenarios", [])

    results = []
    for s in scenarios:
        result = apply_historical_scenario(
            weights, returns,
            s["start"], s["end"], s["name"],
        )
        result["shock_magnitude"] = s.get("shock_magnitude", np.nan)
        results.append(result)

    logger.info(f"  Stress testing: {len(results)} historical scenarios computed")
    return pd.DataFrame(results)


# ── Synthetic Stress Testing ────────────────────────────────────────────────


def apply_synthetic_shock(
    weights: pd.Series,
    cov: pd.DataFrame,
    shocks: dict[str, float],
    scenario_name: str = "synthetic",
    asset_types: pd.Series | None = None,
) -> dict:
    """
    Apply a synthetic shock scenario.

    shocks: dict of shock types and magnitudes, e.g.:
      {"equity_shock": -0.30, "bond_shock": -0.05, "gold_shock": 0.08}

    Applies shocks to appropriate assets based on asset_types.
    """
    common = weights.index.intersection(cov.columns)
    w = weights[common] / weights[common].sum()

    # Default: apply uniform shock
    asset_shocks = pd.Series(0.0, index=common)

    if asset_types is not None:
        at = asset_types[asset_types.index.isin(common)]

        for shock_key, magnitude in shocks.items():
            if "india_equity" in shock_key:
                # India-specific shock — apply to equity assets (heuristic)
                mask = at.isin(["equity_india", "mutual_fund_equity"])
                asset_shocks[mask.index[mask]] = magnitude
            elif "equity" in shock_key:
                mask = at.str.contains("equity", case=False, na=False)
                asset_shocks[mask.index[mask]] = magnitude
            elif "bond" in shock_key:
                mask = at.str.contains("bond|debt|fixed", case=False, na=False)
                asset_shocks[mask.index[mask]] = magnitude
            elif "gold" in shock_key:
                mask = at.str.contains("gold|commodity", case=False, na=False)
                asset_shocks[mask.index[mask]] = magnitude
            elif "tech" in shock_key:
                mask = at.str.contains("tech|nasdaq", case=False, na=False)
                asset_shocks[mask.index[mask]] = magnitude
            elif "fx" in shock_key:
                # FX shock applied as indirect equity impact
                mask = at.str.contains("us_equity|international", case=False, na=False)
                asset_shocks[mask.index[mask]] += magnitude * 0.5
    else:
        # No asset types — apply first numeric shock uniformly
        for v in shocks.values():
            if isinstance(v, (int, float)):
                asset_shocks[:] = v
                break

    # Portfolio impact
    portfolio_impact = float(w @ asset_shocks)

    # Worst-case with correlation spike
    correlation_override = shocks.get("correlation_override")
    vol_multiplier = shocks.get("vol_multiplier", 1.0)

    stressed_vol = np.nan
    if correlation_override is not None:
        vols = np.sqrt(np.diag(cov.loc[common, common].values))
        stressed_cov = np.outer(vols, vols) * correlation_override * vol_multiplier
        np.fill_diagonal(stressed_cov, (vols * vol_multiplier) ** 2)
        stressed_vol = float(np.sqrt(w.values @ stressed_cov @ w.values) * np.sqrt(252))

    return {
        "scenario": scenario_name,
        "portfolio_impact": portfolio_impact,
        "stressed_vol": stressed_vol,
        "asset_shocks": asset_shocks.to_dict(),
        "n_affected": int((asset_shocks != 0).sum()),
    }


def run_synthetic_stress_tests(
    weights: pd.Series,
    cov: pd.DataFrame,
    asset_types: pd.Series | None = None,
) -> pd.DataFrame:
    """Run all configured synthetic scenarios."""
    cfg = _load_config().get("stress_testing", {})
    scenarios = cfg.get("synthetic_scenarios", [])

    results = []
    for s in scenarios:
        name = s.pop("name", "unnamed")
        result = apply_synthetic_shock(weights, cov, s, name, asset_types)
        results.append(result)

    logger.info(f"  Stress testing: {len(results)} synthetic scenarios computed")
    return pd.DataFrame(results)


# ── Combined Stress Report ──────────────────────────────────────────────────


def build_stress_test_report(
    weights: pd.Series,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    asset_types: pd.Series | None = None,
) -> dict:
    """Full stress testing report: historical + synthetic."""
    hist = run_historical_stress_tests(weights, returns)
    synth = run_synthetic_stress_tests(weights, cov, asset_types)

    worst_historical = hist.loc[hist["portfolio_return"].idxmin()] if not hist.empty else {}
    worst_synthetic = synth.loc[synth["portfolio_impact"].idxmin()] if not synth.empty else {}

    return {
        "historical_scenarios": hist.to_dict("records"),
        "synthetic_scenarios": synth.to_dict("records"),
        "worst_historical": worst_historical.to_dict() if len(worst_historical) else {},
        "worst_synthetic": worst_synthetic.to_dict() if len(worst_synthetic) else {},
    }
