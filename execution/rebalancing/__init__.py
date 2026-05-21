"""
Execution Engine — Drift-Based Rebalancing.

Move from fixed calendar rebalancing to event-driven rebalancing.

Rebalance ONLY when:
  - Weight drift exceeds threshold
  - Regime changes materially
  - Risk contribution shifts significantly
  - Confidence drops sharply

Dynamic thresholds: panic → wider (avoid churn), low vol → tighter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/execution_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


def get_drift_threshold(regime: str = "risk_on") -> float:
    """
    Dynamic drift threshold based on current regime.

    Panic → wider (0.08) to avoid churning in chaos.
    Low vol / risk_on → tighter (0.05) to capture drift.
    """
    cfg = _load_config().get("drift", {})
    thresholds = cfg.get("regime_thresholds", {})
    return thresholds.get(regime, cfg.get("base_threshold", 0.05))


def compute_weight_drift(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> dict:
    """
    Compute per-asset and aggregate weight drift.

    Returns: {
        per_asset: {ticker: drift},
        max_drift: float,
        max_drift_ticker: str,
        total_drift: float (one-way turnover),
        n_drifted: int,
    }
    """
    all_tickers = set(current_weights) | set(target_weights)
    per_asset = {}

    for t in all_tickers:
        if t.startswith("_"):
            continue
        curr = current_weights.get(t, 0)
        tgt = target_weights.get(t, 0)
        per_asset[t] = abs(curr - tgt)

    if not per_asset:
        return {
            "per_asset": {},
            "max_drift": 0,
            "max_drift_ticker": "",
            "total_drift": 0,
            "n_drifted": 0,
        }

    max_ticker = max(per_asset, key=per_asset.get)

    return {
        "per_asset": per_asset,
        "max_drift": per_asset[max_ticker],
        "max_drift_ticker": max_ticker,
        "total_drift": sum(per_asset.values()) / 2,
        "n_drifted": sum(1 for v in per_asset.values() if v > 0.01),
    }


def compute_risk_drift(
    current_risk_pct: dict[str, float],
    target_risk_pct: dict[str, float],
) -> dict:
    """Compute drift in risk contributions."""
    all_tickers = set(current_risk_pct) | set(target_risk_pct)
    drifts = {}

    for t in all_tickers:
        curr = current_risk_pct.get(t, 0)
        tgt = target_risk_pct.get(t, 0)
        drifts[t] = abs(curr - tgt)

    max_drift = max(drifts.values()) if drifts else 0

    return {
        "per_asset": drifts,
        "max_risk_drift": max_drift,
        "total_risk_drift": sum(drifts.values()) / 2,
    }


def should_rebalance(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    regime: str = "risk_on",
    regime_changed: bool = False,
    current_risk_pct: dict[str, float] | None = None,
    target_risk_pct: dict[str, float] | None = None,
    confidence: float | None = None,
    prev_confidence: float | None = None,
) -> dict:
    """
    Event-driven rebalance decision.

    Triggers:
      1. Weight drift > regime-adjusted threshold
      2. Regime changed materially
      3. Risk contribution drift > threshold
      4. Confidence dropped sharply

    Returns: {
        should_rebalance: bool,
        trigger: str,
        details: dict,
    }
    """
    cfg = _load_config().get("drift", {})
    risk_thresh = cfg.get("risk_contribution_threshold", 0.10)
    conf_thresh = cfg.get("confidence_drop_threshold", 0.20)

    # 1. Weight drift
    drift_threshold = get_drift_threshold(regime)
    drift = compute_weight_drift(current_weights, target_weights)

    if drift["max_drift"] > drift_threshold:
        return {
            "should_rebalance": True,
            "trigger": "weight_drift",
            "threshold": drift_threshold,
            "drift": drift,
            "details": {
                "max_drift": drift["max_drift"],
                "max_drift_ticker": drift["max_drift_ticker"],
                "regime": regime,
            },
        }

    # 2. Regime change
    if regime_changed and regime in ("panic", "risk_off"):
        return {
            "should_rebalance": True,
            "trigger": "regime_change",
            "threshold": 0,
            "drift": drift,
            "details": {
                "regime": regime,
                "regime_changed": True,
            },
        }

    # 3. Risk contribution drift
    if current_risk_pct and target_risk_pct:
        risk_drift = compute_risk_drift(current_risk_pct, target_risk_pct)
        if risk_drift["max_risk_drift"] > risk_thresh:
            return {
                "should_rebalance": True,
                "trigger": "risk_drift",
                "threshold": risk_thresh,
                "drift": drift,
                "details": risk_drift,
            }

    # 4. Confidence drop
    if confidence is not None and prev_confidence is not None:
        conf_drop = prev_confidence - confidence
        if conf_drop > conf_thresh:
            return {
                "should_rebalance": True,
                "trigger": "confidence_drop",
                "threshold": conf_thresh,
                "drift": drift,
                "details": {
                    "confidence": confidence,
                    "prev_confidence": prev_confidence,
                    "drop": conf_drop,
                },
            }

    return {
        "should_rebalance": False,
        "trigger": "none",
        "threshold": drift_threshold,
        "drift": drift,
        "details": {
            "max_drift": drift["max_drift"],
            "regime": regime,
        },
    }
