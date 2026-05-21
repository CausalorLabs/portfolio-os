"""
Regime Intelligence Engine — context-aware portfolio behavior.

Public API:
    run_regime_pipeline()    — full pipeline: features → detect → persist → evaluate
    get_current_regime()     — latest regime state
    get_regime_behavior()    — portfolio parameter overrides for a regime
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from regimes.behavior import RegimeBehavior, apply_regime_constraints, get_regime_behavior
from regimes.detectors import detect_regimes
from regimes.evaluation import evaluate_crisis_alignment, evaluate_predictive_value, evaluate_regime_quality
from regimes.features import build_regime_features
from regimes.transitions import compute_regime_durations, compute_stability_metrics, compute_transition_matrix

PROCESSED = Path("data/processed")

__all__ = [
    "run_regime_pipeline",
    "get_current_regime",
    "get_regime_behavior",
    "apply_regime_constraints",
    "RegimeBehavior",
    "build_regime_features",
    "detect_regimes",
    "compute_transition_matrix",
    "compute_regime_durations",
    "compute_stability_metrics",
    "evaluate_regime_quality",
    "evaluate_predictive_value",
    "evaluate_crisis_alignment",
]


def run_regime_pipeline(
    inr_prices: pd.DataFrame | None = None,
    nav_series: pd.DataFrame | None = None,
    save: bool = True,
) -> dict:
    """
    Full regime intelligence pipeline.

    Steps:
        1. Build regime features (VIX, momentum, breadth, correlation, vol, liquidity)
        2. Detect regimes (rule-based V1 with persistence)
        3. Compute transition matrix and stability metrics
        4. Evaluate regime quality (predictive value, crisis alignment)
        5. Save outputs to data/processed/

    Args:
        inr_prices: Long-format price data. Loaded from parquet if None.
        nav_series: Portfolio NAV data. Loaded from parquet if None.
        save: Whether to save outputs to parquet.

    Returns:
        Dict with keys: regimes, features, transition_matrix, stability,
                        quality_score, current_regime, behavior
    """
    logger.info("=" * 60)
    logger.info("REGIME INTELLIGENCE PIPELINE")
    logger.info("=" * 60)

    # 1. Features
    logger.info("\n▸ Step 1 — Regime Feature Pipeline")
    features = build_regime_features(inr_prices)
    logger.info(f"  Features: {len(features)} days, {len(features.columns) - 1} indicators")

    # 2. Detection
    logger.info("\n▸ Step 2 — Regime Detection (rule-based V1)")
    regimes = detect_regimes(features)

    # 3. Transitions
    logger.info("\n▸ Step 3 — Transition Analysis")
    transition_matrix = compute_transition_matrix(regimes["regime"])
    stability = compute_stability_metrics(regimes)
    durations = compute_regime_durations(regimes)

    logger.info(f"  Transitions/year: {stability.get('transitions_per_year', 'N/A')}")
    logger.info(f"  Avg duration: {stability.get('avg_duration_days', 'N/A')} days")
    logger.info(f"  Dominant: {stability.get('dominant_regime', 'N/A')} ({stability.get('dominant_pct', 0):.0f}%)")

    # 4. Evaluation
    logger.info("\n▸ Step 4 — Regime Quality Evaluation")
    quality_score = {}
    if nav_series is None:
        nav_path = PROCESSED / "portfolio_nav.parquet"
        if nav_path.exists():
            nav_series = pd.read_parquet(nav_path)
            nav_series["date"] = pd.to_datetime(nav_series["date"])

    if nav_series is not None:
        quality_score = evaluate_regime_quality(regimes, nav_series)

    # 5. Current regime
    current = regimes.iloc[-1]
    current_regime = current["regime"]
    behavior = get_regime_behavior(current_regime)

    logger.info(f"\n  Current regime: {current_regime.upper()} (confidence={current['confidence']:.2f})")
    logger.info(f"  Behavior: equity≤{behavior.max_equity_weight:.0%}, "
                f"drift={behavior.rebalance_drift_threshold:.0%}, "
                f"tilt={behavior.tilt_strength}")

    # 6. Save
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)

        features.to_parquet(PROCESSED / "regime_features.parquet", index=False)
        regimes.to_parquet(PROCESSED / "regime_states.parquet", index=False)
        durations.to_parquet(PROCESSED / "regime_durations.parquet", index=False)
        transition_matrix.to_csv(PROCESSED / "regime_transitions.csv")
        logger.info("  Saved: regime_features, regime_states, regime_durations, regime_transitions")

    return {
        "regimes": regimes,
        "features": features,
        "transition_matrix": transition_matrix,
        "stability": stability,
        "durations": durations,
        "quality_score": quality_score,
        "current_regime": current_regime,
        "behavior": behavior,
    }


def get_current_regime() -> tuple[str, RegimeBehavior]:
    """
    Get the current regime and its portfolio behavior.

    Loads from saved parquet if available, otherwise runs detection.
    """
    states_path = PROCESSED / "regime_states.parquet"
    if states_path.exists():
        regimes = pd.read_parquet(states_path)
        if not regimes.empty:
            current = regimes.iloc[-1]["regime"]
            return current, get_regime_behavior(current)

    # Fallback: run pipeline
    result = run_regime_pipeline()
    return result["current_regime"], result["behavior"]
"""
Feature store — re-exports from features/ package.

This module provides the MVP interface to the feature pipeline.
Delegates to the POC feature modules under features/.
"""

from features.feature_store import build_feature_store, load_feature_store, save_feature_store
from features.signal_ranker import calculate_composite_score
from features.validators import validate_features

__all__ = [
    "build_feature_store",
    "save_feature_store",
    "load_feature_store",
    "calculate_composite_score",
    "validate_features",
]
