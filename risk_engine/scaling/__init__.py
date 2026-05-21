"""
Risk Engine — Volatility Targeting & Risk Scaling.

Dynamic position sizing: when vol is high, reduce exposure.
When vol is low, deploy more capital.

This is the bridge between risk measurement and portfolio action.
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


# ── Volatility Targeting ────────────────────────────────────────────────────


def compute_vol_scaling_factor(
    realized_vol: float,
    target_vol: float | None = None,
    min_scaling: float | None = None,
    max_scaling: float | None = None,
) -> float:
    """
    Compute how much to scale exposure to hit target vol.

    scaling = target_vol / realized_vol, clamped to [min, max].
    """
    cfg = _load_config().get("volatility_targeting", {})

    if target_vol is None:
        target_vol = cfg.get("target_portfolio_vol", 0.12)
    if min_scaling is None:
        min_scaling = cfg.get("min_scaling", 0.50)
    if max_scaling is None:
        max_scaling = cfg.get("max_scaling", 1.20)

    if realized_vol < 1e-6:
        return max_scaling

    raw_scaling = target_vol / realized_vol
    return float(np.clip(raw_scaling, min_scaling, max_scaling))


def apply_vol_scaling(
    weights: pd.Series,
    realized_vol: float,
    target_vol: float | None = None,
) -> tuple[pd.Series, float]:
    """
    Scale portfolio weights to target volatility.

    Returns (scaled_weights, scaling_factor).
    Excess (1 - sum(scaled)) goes to cash.
    """
    factor = compute_vol_scaling_factor(realized_vol, target_vol)

    scaled = weights * factor
    cash = max(0, 1.0 - scaled.sum())

    logger.info(
        f"  Vol scaling: realized={realized_vol:.2%}, "
        f"factor={factor:.2f}, cash={cash:.2%}"
    )

    return scaled, factor


# ── Smoothed Scaling ────────────────────────────────────────────────────────


def compute_smoothed_scaling(
    vol_series: pd.Series,
    target_vol: float | None = None,
    halflife: int | None = None,
) -> pd.Series:
    """
    EMA-smoothed scaling factor to avoid whipsawing.

    Prevents aggressive daily rebalancing. Uses exponential smoothing
    on the raw scaling factor.
    """
    cfg = _load_config().get("volatility_targeting", {})

    if target_vol is None:
        target_vol = cfg.get("target_portfolio_vol", 0.12)
    if halflife is None:
        halflife = cfg.get("smoothing_halflife", 10)

    min_s = cfg.get("min_scaling", 0.50)
    max_s = cfg.get("max_scaling", 1.20)

    raw_scaling = target_vol / vol_series.clip(lower=1e-6)
    raw_scaling = raw_scaling.clip(lower=min_s, upper=max_s)

    smoothed = raw_scaling.ewm(halflife=halflife, min_periods=3).mean()

    return smoothed


# ── Risk-Scaled Returns ────────────────────────────────────────────────────


def compute_risk_scaled_returns(
    returns: pd.Series,
    vol_series: pd.Series,
    target_vol: float | None = None,
) -> pd.Series:
    """
    Compute what returns would have been with vol targeting.

    Useful for backtest analysis.
    """
    scaling = compute_smoothed_scaling(vol_series, target_vol)

    # Align
    common = returns.index.intersection(scaling.index)
    return returns[common] * scaling[common]


# ── Risk Scaling Report ────────────────────────────────────────────────────


def build_scaling_report(
    current_vol: float,
    target_vol: float | None = None,
) -> dict:
    """Snapshot of current risk scaling state."""
    cfg = _load_config().get("volatility_targeting", {})
    if target_vol is None:
        target_vol = cfg.get("target_portfolio_vol", 0.12)

    factor = compute_vol_scaling_factor(current_vol, target_vol)
    cash_allocation = max(0, 1.0 - factor)

    return {
        "current_vol": current_vol,
        "target_vol": target_vol,
        "scaling_factor": factor,
        "implied_cash": cash_allocation,
        "is_scaling_active": factor < 1.0,
        "recommendation": (
            "Reduce exposure" if factor < 0.9
            else "Increase exposure" if factor > 1.05
            else "Hold steady"
        ),
    }
