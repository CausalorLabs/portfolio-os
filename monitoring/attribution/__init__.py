"""
Monitoring — Performance & Factor Attribution Engine.

Explains WHERE returns came from:
  - Allocation effect (sector/country/asset class overweights)
  - Selection effect (picking better assets within categories)
  - Timing effect (regime shifts, vol scaling, defensive positioning)
  - Currency effect (FX contribution for cross-border portfolios)
  - Interaction effect (residual)

Factor decomposition:
  - Market beta, momentum, low volatility, quality, size
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/monitoring.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Performance Attribution (Brinson-Hood-Beebower)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class AttributionResult:
    """Full attribution breakdown for a period."""
    period_start: date
    period_end: date
    portfolio_return: float
    benchmark_return: float
    active_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    timing_effect: float
    currency_effect: float
    total_explained: float
    residual: float
    sector_details: list[dict] = field(default_factory=list)


def calculate_allocation_effect(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    benchmark_returns: dict[str, float],
    benchmark_total_return: float,
) -> dict[str, float]:
    """
    Brinson allocation effect per sector/asset.

    Allocation = Σ (w_p,i - w_b,i) × (R_b,i - R_b)

    Measures value added from overweighting sectors that outperform.
    """
    effects = {}
    all_sectors = set(portfolio_weights) | set(benchmark_weights)

    for sector in all_sectors:
        w_p = portfolio_weights.get(sector, 0.0)
        w_b = benchmark_weights.get(sector, 0.0)
        r_b = benchmark_returns.get(sector, 0.0)
        effects[sector] = (w_p - w_b) * (r_b - benchmark_total_return)

    return effects


def calculate_selection_effect(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
) -> dict[str, float]:
    """
    Brinson selection effect per sector/asset.

    Selection = Σ w_b,i × (R_p,i - R_b,i)

    Measures value added from picking better assets within categories.
    """
    effects = {}
    all_sectors = set(portfolio_weights) | set(benchmark_weights)

    for sector in all_sectors:
        w_b = benchmark_weights.get(sector, 0.0)
        r_p = portfolio_returns.get(sector, 0.0)
        r_b = benchmark_returns.get(sector, 0.0)
        effects[sector] = w_b * (r_p - r_b)

    return effects


def calculate_interaction_effect(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
) -> dict[str, float]:
    """
    Brinson interaction effect per sector/asset.

    Interaction = Σ (w_p,i - w_b,i) × (R_p,i - R_b,i)

    Cross-term: overweighting sectors where we also pick better.
    """
    effects = {}
    all_sectors = set(portfolio_weights) | set(benchmark_weights)

    for sector in all_sectors:
        w_p = portfolio_weights.get(sector, 0.0)
        w_b = benchmark_weights.get(sector, 0.0)
        r_p = portfolio_returns.get(sector, 0.0)
        r_b = benchmark_returns.get(sector, 0.0)
        effects[sector] = (w_p - w_b) * (r_p - r_b)

    return effects


def calculate_timing_effect(
    regime_returns: dict[str, float],
    regime_weights: dict[str, dict[str, float]],
    baseline_weights: dict[str, float],
    asset_returns: dict[str, float],
) -> float:
    """
    Timing effect from regime-aware allocation changes.

    Measures value added from adjusting weights based on regime shifts.
    Timing = Σ (regime_weight_i - baseline_weight_i) × return_i
    """
    timing = 0.0
    for asset, base_w in baseline_weights.items():
        r = asset_returns.get(asset, 0.0)
        for regime, weights in regime_weights.items():
            regime_w = weights.get(asset, base_w)
            timing += (regime_w - base_w) * r

    return timing


def calculate_currency_effect(
    portfolio_weights: dict[str, float],
    local_returns: dict[str, float],
    total_returns: dict[str, float],
) -> dict[str, float]:
    """
    Currency contribution to performance.

    Currency effect = Σ w_i × (total_return_i - local_return_i)
    """
    effects = {}
    for asset, weight in portfolio_weights.items():
        local_r = local_returns.get(asset, 0.0)
        total_r = total_returns.get(asset, 0.0)
        effects[asset] = weight * (total_r - local_r)

    return effects


def run_performance_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
    period_start: date | None = None,
    period_end: date | None = None,
    local_returns: dict[str, float] | None = None,
    total_returns: dict[str, float] | None = None,
    regime_weights: dict[str, dict[str, float]] | None = None,
    baseline_weights: dict[str, float] | None = None,
) -> AttributionResult:
    """
    Full Brinson-Hood-Beebower attribution with timing and currency.

    Returns complete breakdown of active return sources.
    """
    # Portfolio and benchmark total returns
    port_total = sum(
        portfolio_weights.get(a, 0) * portfolio_returns.get(a, 0)
        for a in set(portfolio_weights) | set(portfolio_returns)
    )
    bench_total = sum(
        benchmark_weights.get(a, 0) * benchmark_returns.get(a, 0)
        for a in set(benchmark_weights) | set(benchmark_returns)
    )
    active = port_total - bench_total

    # Brinson decomposition
    alloc = calculate_allocation_effect(
        portfolio_weights, benchmark_weights,
        benchmark_returns, bench_total,
    )
    select = calculate_selection_effect(
        portfolio_weights, benchmark_weights,
        portfolio_returns, benchmark_returns,
    )
    interact = calculate_interaction_effect(
        portfolio_weights, benchmark_weights,
        portfolio_returns, benchmark_returns,
    )

    alloc_total = sum(alloc.values())
    select_total = sum(select.values())
    interact_total = sum(interact.values())

    # Timing effect
    timing_total = 0.0
    if regime_weights and baseline_weights:
        timing_total = calculate_timing_effect(
            {}, regime_weights, baseline_weights, portfolio_returns,
        )

    # Currency effect
    currency_total = 0.0
    currency_details = {}
    if local_returns and total_returns:
        currency_details = calculate_currency_effect(
            portfolio_weights, local_returns, total_returns,
        )
        currency_total = sum(currency_details.values())

    # Build sector details
    sector_details = []
    all_assets = set(portfolio_weights) | set(benchmark_weights)
    for asset in sorted(all_assets):
        sector_details.append({
            "asset": asset,
            "portfolio_weight": portfolio_weights.get(asset, 0),
            "benchmark_weight": benchmark_weights.get(asset, 0),
            "portfolio_return": portfolio_returns.get(asset, 0),
            "benchmark_return": benchmark_returns.get(asset, 0),
            "allocation_effect": alloc.get(asset, 0),
            "selection_effect": select.get(asset, 0),
            "interaction_effect": interact.get(asset, 0),
            "currency_effect": currency_details.get(asset, 0),
        })

    explained = alloc_total + select_total + interact_total + timing_total + currency_total
    residual = active - explained

    result = AttributionResult(
        period_start=period_start or date.today(),
        period_end=period_end or date.today(),
        portfolio_return=port_total,
        benchmark_return=bench_total,
        active_return=active,
        allocation_effect=alloc_total,
        selection_effect=select_total,
        interaction_effect=interact_total,
        timing_effect=timing_total,
        currency_effect=currency_total,
        total_explained=explained,
        residual=residual,
        sector_details=sector_details,
    )

    logger.info(
        f"  Attribution: active={active:.4f} "
        f"(alloc={alloc_total:.4f}, select={select_total:.4f}, "
        f"interact={interact_total:.4f}, timing={timing_total:.4f}, "
        f"currency={currency_total:.4f})"
    )
    return result


def attribution_to_dataframe(result: AttributionResult) -> pd.DataFrame:
    """Convert attribution result to DataFrame."""
    return pd.DataFrame(result.sector_details)


def attribution_summary_series(result: AttributionResult) -> pd.Series:
    """Summary series of attribution effects."""
    return pd.Series({
        "portfolio_return": result.portfolio_return,
        "benchmark_return": result.benchmark_return,
        "active_return": result.active_return,
        "allocation_effect": result.allocation_effect,
        "selection_effect": result.selection_effect,
        "interaction_effect": result.interaction_effect,
        "timing_effect": result.timing_effect,
        "currency_effect": result.currency_effect,
        "total_explained": result.total_explained,
        "residual": result.residual,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Factor Attribution
# ══════════════════════════════════════════════════════════════════════════════


FACTOR_DEFINITIONS = {
    "market_beta": {
        "description": "General equity market exposure",
        "proxy_features": ["return_21d"],
    },
    "momentum": {
        "description": "Trend-following exposure",
        "proxy_features": ["momentum_63d", "momentum_126d"],
    },
    "low_volatility": {
        "description": "Defensive / low-risk exposure",
        "proxy_features": ["volatility_21d"],
        "invert": True,  # Low vol → positive factor
    },
    "quality": {
        "description": "Stability and profitability proxies",
        "proxy_features": ["sharpe_rolling"],
    },
    "size": {
        "description": "Size tilt (market cap proxy)",
        "proxy_features": ["avg_volume_ratio"],
    },
}


@dataclass
class FactorAttributionResult:
    """Factor decomposition of portfolio returns."""
    factors: list[dict]
    total_factor_return: float
    specific_return: float
    r_squared: float


def compute_factor_exposures(
    weights: dict[str, float],
    feature_data: pd.DataFrame,
    factors: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute portfolio-level factor exposures.

    Exposure_f = Σ w_i × z_i,f  (weighted average of z-scored features)
    """
    cfg = _load_config().get("attribution", {})
    if factors is None:
        factors = cfg.get("factors", list(FACTOR_DEFINITIONS.keys()))

    exposures = {}
    for factor in factors:
        fdef = FACTOR_DEFINITIONS.get(factor)
        if fdef is None:
            continue

        proxy_cols = fdef.get("proxy_features", [])
        available_cols = [c for c in proxy_cols if c in feature_data.columns]
        if not available_cols:
            exposures[factor] = 0.0
            continue

        # Z-score the features
        factor_vals = feature_data[available_cols].mean(axis=1)
        if factor_vals.std() > 0:
            z_scored = (factor_vals - factor_vals.mean()) / factor_vals.std()
        else:
            z_scored = factor_vals * 0

        # Weighted portfolio exposure
        exposure = 0.0
        for ticker, w in weights.items():
            if ticker in z_scored.index:
                val = z_scored.loc[ticker]
                if fdef.get("invert", False):
                    val = -val
                exposure += w * val

        exposures[factor] = float(exposure)

    return exposures


def compute_factor_returns(
    returns: pd.DataFrame,
    feature_data: pd.DataFrame,
    factors: list[str] | None = None,
) -> dict[str, float]:
    """
    Estimate factor returns using cross-sectional regression.

    For each factor, compute the return of a long-short factor portfolio.
    Top quintile vs bottom quintile.
    """
    cfg = _load_config().get("attribution", {})
    if factors is None:
        factors = cfg.get("factors", list(FACTOR_DEFINITIONS.keys()))

    factor_returns = {}
    for factor in factors:
        fdef = FACTOR_DEFINITIONS.get(factor)
        if fdef is None:
            continue

        proxy_cols = fdef.get("proxy_features", [])
        available_cols = [c for c in proxy_cols if c in feature_data.columns]
        if not available_cols or returns.empty:
            factor_returns[factor] = 0.0
            continue

        # Use mean of proxy features as factor score
        scores = feature_data[available_cols].mean(axis=1)
        if fdef.get("invert", False):
            scores = -scores

        # Quintile long-short
        common = scores.index.intersection(returns.index)
        if len(common) < 5:
            factor_returns[factor] = 0.0
            continue

        scores_common = scores.loc[common]
        returns_common = returns.loc[common]

        top_q = scores_common.quantile(0.8)
        bot_q = scores_common.quantile(0.2)

        top_mask = scores_common >= top_q
        bot_mask = scores_common <= bot_q

        if top_mask.sum() > 0 and bot_mask.sum() > 0:
            # Handle both Series and DataFrame returns
            if isinstance(returns_common, pd.DataFrame):
                top_ret = returns_common.loc[top_mask].mean().mean()
                bot_ret = returns_common.loc[bot_mask].mean().mean()
            else:
                top_ret = returns_common.loc[top_mask].mean()
                bot_ret = returns_common.loc[bot_mask].mean()
            factor_returns[factor] = float(top_ret - bot_ret)
        else:
            factor_returns[factor] = 0.0

    return factor_returns


def run_factor_attribution(
    weights: dict[str, float],
    asset_returns: dict[str, float],
    feature_data: pd.DataFrame,
    returns_data: pd.DataFrame | None = None,
    factors: list[str] | None = None,
) -> FactorAttributionResult:
    """
    Full factor attribution: exposure × factor_return = contribution.

    Returns per-factor breakdown and specific (unexplained) return.
    """
    exposures = compute_factor_exposures(weights, feature_data, factors)

    # Estimate factor returns if we have cross-sectional return data
    if returns_data is not None and not returns_data.empty:
        factor_rets = compute_factor_returns(returns_data, feature_data, factors)
    else:
        # Use simple proxy: factor exposure × portfolio return
        port_return = sum(weights.get(a, 0) * asset_returns.get(a, 0)
                          for a in set(weights) | set(asset_returns))
        factor_rets = {f: port_return * 0.1 for f in exposures}

    # Compute contributions
    factor_details = []
    total_factor = 0.0
    for factor, exposure in exposures.items():
        f_return = factor_rets.get(factor, 0.0)
        contribution = exposure * f_return
        total_factor += contribution

        factor_details.append({
            "factor": factor,
            "exposure": round(exposure, 4),
            "factor_return": round(f_return, 6),
            "contribution": round(contribution, 6),
            "description": FACTOR_DEFINITIONS.get(factor, {}).get("description", ""),
        })

    # Specific return
    port_return = sum(weights.get(a, 0) * asset_returns.get(a, 0)
                      for a in set(weights) | set(asset_returns))
    specific = port_return - total_factor

    # R-squared (how much is explained by factors)
    r_sq = (total_factor / port_return) if abs(port_return) > 1e-10 else 0.0
    r_sq = max(0.0, min(1.0, abs(r_sq)))

    result = FactorAttributionResult(
        factors=factor_details,
        total_factor_return=round(total_factor, 6),
        specific_return=round(specific, 6),
        r_squared=round(r_sq, 4),
    )

    logger.info(
        f"  Factor attribution: {len(factor_details)} factors, "
        f"R²={r_sq:.2%}, specific={specific:.4f}"
    )
    return result


def factor_attribution_to_dataframe(result: FactorAttributionResult) -> pd.DataFrame:
    """Convert factor attribution to DataFrame."""
    return pd.DataFrame(result.factors)
