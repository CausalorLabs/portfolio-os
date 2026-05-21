"""
Portfolio behavior mapping — connect regime state to portfolio parameters.

Given a regime, returns the parameter overrides that should be applied
to optimization, covariance, rebalancing, and tilt logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf


@dataclass
class RegimeBehavior:
    """Portfolio behavior parameters for a given regime."""
    regime: str
    max_equity_weight: float
    covariance_method: str
    covariance_ewma_halflife: int
    rebalance_drift_threshold: float
    tilt_strength: float


# Default behaviors (used if config unavailable)
_DEFAULTS: dict[str, dict] = {
    "risk_on": {
        "max_equity_weight": 0.80,
        "covariance_method": "shrinkage",
        "covariance_ewma_halflife": 120,
        "rebalance_drift_threshold": 0.05,
        "tilt_strength": 0.25,
    },
    "risk_off": {
        "max_equity_weight": 0.55,
        "covariance_method": "shrinkage",
        "covariance_ewma_halflife": 60,
        "rebalance_drift_threshold": 0.04,
        "tilt_strength": 0.10,
    },
    "panic": {
        "max_equity_weight": 0.35,
        "covariance_method": "ewma",
        "covariance_ewma_halflife": 30,
        "rebalance_drift_threshold": 0.08,
        "tilt_strength": 0.0,
    },
    "high_vol": {
        "max_equity_weight": 0.60,
        "covariance_method": "ewma",
        "covariance_ewma_halflife": 45,
        "rebalance_drift_threshold": 0.06,
        "tilt_strength": 0.15,
    },
}


def _load_behavior_config() -> dict[str, dict]:
    """Load behavior mapping from configs/regimes.yaml."""
    path = Path("configs/regimes.yaml")
    if path.exists():
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cfg.get("behavior", _DEFAULTS)
    return _DEFAULTS


def get_regime_behavior(regime: str) -> RegimeBehavior:
    """
    Get portfolio behavior parameters for the given regime.

    Args:
        regime: One of "risk_on", "risk_off", "panic", "high_vol"

    Returns:
        RegimeBehavior with all override parameters.
    """
    behaviors = _load_behavior_config()
    params = behaviors.get(regime, _DEFAULTS.get("risk_on", {}))

    behavior = RegimeBehavior(
        regime=regime,
        max_equity_weight=params.get("max_equity_weight", 0.70),
        covariance_method=params.get("covariance_method", "shrinkage"),
        covariance_ewma_halflife=params.get("covariance_ewma_halflife", 60),
        rebalance_drift_threshold=params.get("rebalance_drift_threshold", 0.05),
        tilt_strength=params.get("tilt_strength", 0.20),
    )

    logger.debug(
        f"Regime behavior [{regime}]: equity≤{behavior.max_equity_weight:.0%}, "
        f"cov={behavior.covariance_method}, drift={behavior.rebalance_drift_threshold:.0%}, "
        f"tilt={behavior.tilt_strength}"
    )

    return behavior


def apply_regime_constraints(
    weights: dict[str, float],
    behavior: RegimeBehavior,
    asset_types: dict[str, str],
) -> dict[str, float]:
    """
    Apply regime-based constraints to portfolio weights.

    Caps total equity exposure at behavior.max_equity_weight,
    redistributing excess pro-rata to non-equity assets.

    Args:
        weights: Current target weights {ticker: weight}
        behavior: RegimeBehavior from get_regime_behavior()
        asset_types: {ticker: asset_type} mapping

    Returns:
        Adjusted weights dict.
    """
    equity_types = {"equity", "etf"}
    equity_tickers = {t for t, atype in asset_types.items() if atype in equity_types and t in weights}
    non_equity_tickers = {t for t in weights if t not in equity_tickers}

    equity_total = sum(weights.get(t, 0) for t in equity_tickers)
    max_eq = behavior.max_equity_weight

    if equity_total <= max_eq or not non_equity_tickers:
        return weights

    # Scale down equity pro-rata
    scale = max_eq / equity_total
    excess = equity_total - max_eq

    adjusted = {}
    non_eq_total = sum(weights.get(t, 0) for t in non_equity_tickers)

    for t in equity_tickers:
        adjusted[t] = weights[t] * scale

    # Redistribute excess to non-equity pro-rata
    for t in non_equity_tickers:
        share = (weights[t] / non_eq_total) if non_eq_total > 0 else (1.0 / len(non_equity_tickers))
        adjusted[t] = weights[t] + excess * share

    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {t: w / total for t, w in adjusted.items()}

    logger.info(
        f"Regime constraint [{behavior.regime}]: equity {equity_total:.1%} → {max_eq:.1%} "
        f"(redistributed {excess:.1%} to {len(non_equity_tickers)} non-equity)"
    )

    return adjusted
