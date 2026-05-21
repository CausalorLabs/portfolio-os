"""
Rule-based regime detector (V1) — interpretable, deterministic.

Classifies each day into one of: risk_on, risk_off, panic, high_vol.
Includes hysteresis and persistence logic to prevent regime flapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_regime_config() -> dict:
    """Load regime config from configs/regimes.yaml."""
    path = Path("configs/regimes.yaml")
    if path.exists():
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    # Hardcoded defaults if config missing
    return {
        "detector": {
            "panic": {"min_conditions": 2, "vix_above": 30.0, "vol_spike_ratio": 1.8, "breadth_below": 0.35},
            "risk_off": {"min_conditions": 2, "momentum_below": -0.02, "vol_above_median": True, "breadth_below": 0.45},
            "risk_on": {"min_conditions": 2, "momentum_above": 0.05, "breadth_above": 0.55, "vol_below_median": True},
            "high_vol": {"vol_spike_ratio": 1.3, "vix_above": 20.0},
        },
        "persistence": {"min_regime_days": 5, "confirmation_window": 3, "hysteresis_factor": 0.8},
    }


# ── Raw classification (no persistence) ─────────────────────────────────────


def _classify_row(row: pd.Series, cfg: dict, vol_median: float) -> tuple[str, float]:
    """
    Classify a single day into a regime.

    Returns (regime, confidence) where confidence is 0-1.
    Priority order: panic > risk_off > high_vol > risk_on.
    """
    det = cfg["detector"]
    vix = row.get("vix", 18.0)
    vix_zscore = row.get("vix_zscore", 0.0)
    spy_mom = row.get("spy_momentum", 0.0)
    nifty_mom = row.get("nifty_momentum", 0.0)
    breadth = row.get("breadth_score", 0.5)
    vol_ratio = row.get("vol_regime_ratio", 1.0)
    realized_vol = row.get("realized_vol", 0.15)
    cross_corr = row.get("cross_asset_corr", 0.3)
    liq_stress = row.get("liquidity_stress", 0.3)

    avg_momentum = np.nanmean([spy_mom, nifty_mom])

    # ── Panic check ──────────────────────────────────────────────────────
    panic_cfg = det["panic"]
    panic_signals = 0
    if vix > panic_cfg["vix_above"]:
        panic_signals += 1
    if vol_ratio > panic_cfg["vol_spike_ratio"]:
        panic_signals += 1
    if breadth < panic_cfg["breadth_below"]:
        panic_signals += 1
    if cross_corr > 0.70:  # correlation spike
        panic_signals += 1

    if panic_signals >= panic_cfg["min_conditions"]:
        confidence = min(1.0, panic_signals / 4)
        return "panic", confidence

    # ── Risk OFF check ───────────────────────────────────────────────────
    roff_cfg = det["risk_off"]
    roff_signals = 0
    if avg_momentum < roff_cfg["momentum_below"]:
        roff_signals += 1
    if realized_vol > vol_median:
        roff_signals += 1
    if breadth < roff_cfg["breadth_below"]:
        roff_signals += 1
    if vix_zscore > 0.5:
        roff_signals += 1

    if roff_signals >= roff_cfg["min_conditions"]:
        confidence = min(1.0, roff_signals / 4)
        return "risk_off", confidence

    # ── High Vol check ───────────────────────────────────────────────────
    hvol_cfg = det["high_vol"]
    if vol_ratio > hvol_cfg["vol_spike_ratio"] and vix > hvol_cfg["vix_above"]:
        confidence = min(1.0, (vol_ratio - 1.0) / 0.5)
        return "high_vol", confidence

    # ── Risk ON (default with conditions) ────────────────────────────────
    ron_cfg = det["risk_on"]
    ron_signals = 0
    if avg_momentum > ron_cfg["momentum_above"]:
        ron_signals += 1
    if breadth > ron_cfg["breadth_above"]:
        ron_signals += 1
    if realized_vol < vol_median:
        ron_signals += 1

    if ron_signals >= ron_cfg["min_conditions"]:
        confidence = min(1.0, ron_signals / 3)
        return "risk_on", confidence

    # ── Fallback: use momentum direction ─────────────────────────────────
    if avg_momentum > 0:
        return "risk_on", 0.3
    else:
        return "risk_off", 0.3


# ── Persistence logic ────────────────────────────────────────────────────────


def _apply_persistence(
    raw_regimes: pd.DataFrame,
    min_regime_days: int = 5,
    confirmation_window: int = 3,
) -> pd.DataFrame:
    """
    Apply anti-flapping persistence rules.

    1. A new regime must be confirmed for `confirmation_window` consecutive days
       before it becomes the active regime.
    2. Once active, a regime holds for at least `min_regime_days`.
    3. Panic always overrides immediately (no confirmation needed).
    """
    result = raw_regimes.copy()
    n = len(result)

    if n == 0:
        return result

    stable_regime = result.iloc[0]["raw_regime"]
    stable_since = 0
    candidate = None
    candidate_count = 0

    stable_regimes = []
    transition_scores = []

    for i in range(n):
        raw = result.iloc[i]["raw_regime"]

        # Panic always overrides immediately
        if raw == "panic":
            stable_regime = "panic"
            stable_since = i
            candidate = None
            candidate_count = 0
            stable_regimes.append("panic")
            transition_scores.append(1.0)
            continue

        # Check if we're still in minimum hold period
        days_in_regime = i - stable_since

        if raw == stable_regime:
            # Same regime — reset candidate
            candidate = None
            candidate_count = 0
            stable_regimes.append(stable_regime)
            transition_scores.append(0.0)
        elif days_in_regime < min_regime_days:
            # Still in minimum hold — ignore the switch signal
            stable_regimes.append(stable_regime)
            transition_scores.append(0.0)
        else:
            # Beyond minimum hold — check confirmation
            if raw == candidate:
                candidate_count += 1
            else:
                candidate = raw
                candidate_count = 1

            if candidate_count >= confirmation_window:
                # Confirmed transition
                stable_regime = candidate
                stable_since = i
                candidate = None
                candidate_count = 0
                stable_regimes.append(stable_regime)
                transition_scores.append(1.0)
            else:
                # Not confirmed yet — hold old regime
                stable_regimes.append(stable_regime)
                transition_scores.append(candidate_count / confirmation_window)

    result["regime"] = stable_regimes
    result["transition_score"] = transition_scores

    return result


# ── Public API ───────────────────────────────────────────────────────────────


def detect_regimes(
    regime_features: pd.DataFrame,
    apply_persistence: bool = True,
) -> pd.DataFrame:
    """
    Classify each day into a market regime.

    Args:
        regime_features: Output of build_regime_features()
        apply_persistence: Whether to apply anti-flapping rules

    Returns:
        DataFrame with: date, regime, confidence, transition_score,
                        raw_regime (pre-persistence)
    """
    cfg = _load_regime_config()
    persistence_cfg = cfg.get("persistence", {})

    # Compute vol median for relative thresholds
    vol_median = regime_features["realized_vol"].median() if "realized_vol" in regime_features.columns else 0.15

    logger.info(f"Detecting regimes over {len(regime_features)} days (vol_median={vol_median:.4f})")

    raw_regimes = []
    confidences = []

    for _, row in regime_features.iterrows():
        regime, confidence = _classify_row(row, cfg, vol_median)
        raw_regimes.append(regime)
        confidences.append(confidence)

    result = pd.DataFrame({
        "date": regime_features["date"].values,
        "raw_regime": raw_regimes,
        "confidence": confidences,
    })

    if apply_persistence:
        result = _apply_persistence(
            result,
            min_regime_days=persistence_cfg.get("min_regime_days", 5),
            confirmation_window=persistence_cfg.get("confirmation_window", 3),
        )
    else:
        result["regime"] = result["raw_regime"]
        result["transition_score"] = 0.0

    # Log distribution
    dist = result["regime"].value_counts()
    logger.info(f"  Regime distribution: {dict(dist)}")

    # Count transitions
    transitions = (result["regime"] != result["regime"].shift()).sum() - 1
    logger.info(f"  Total transitions: {transitions}")

    return result[["date", "regime", "confidence", "transition_score", "raw_regime"]]
