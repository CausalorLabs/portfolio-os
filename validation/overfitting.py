"""
Overfitting detection — identify false alpha from unstable strategies.

Indicators:
  - Very high in-sample Sharpe (suspicious)
  - Large train/test degradation
  - Unstable parameters (high sensitivity)
  - Regime collapse (works only in one regime)
  - Excessive turnover (chasing noise)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def detect_overfitting(
    walkforward_results: pd.DataFrame,
    regime_results: pd.DataFrame | None = None,
    sensitivity_results: pd.DataFrame | None = None,
) -> dict:
    """
    Run overfitting detection battery.

    Parameters
    ----------
    walkforward_results : pd.DataFrame
        Output from run_walkforward_validation().
    regime_results : pd.DataFrame, optional
        Output from evaluate_regime_performance().
    sensitivity_results : pd.DataFrame, optional
        Output from run_parameter_sensitivity().

    Returns
    -------
    dict
        Diagnostic flags, scores, and recommendations.
    """
    flags = []
    scores = {}

    # ── 1. Walk-forward degradation ──────────────────────────────────────
    if not walkforward_results.empty:
        avg_train_sharpe = walkforward_results["train_sharpe"].mean()
        avg_test_sharpe = walkforward_results["test_sharpe"].mean()
        avg_degradation = walkforward_results["sharpe_degradation"].mean()
        consistency = (walkforward_results["test_sharpe"] > 0).mean()

        scores["train_sharpe"] = avg_train_sharpe
        scores["test_sharpe"] = avg_test_sharpe
        scores["sharpe_degradation"] = avg_degradation
        scores["oos_consistency"] = consistency

        # Flag: suspiciously high in-sample Sharpe
        if avg_train_sharpe > 2.5:
            flags.append({
                "flag": "SUSPICIOUS_IS_SHARPE",
                "severity": "high",
                "detail": f"In-sample Sharpe ({avg_train_sharpe:.2f}) is suspiciously high",
            })

        # Flag: large degradation
        if avg_degradation < -0.5:
            flags.append({
                "flag": "LARGE_OOS_DEGRADATION",
                "severity": "high",
                "detail": f"Avg Sharpe degradation ({avg_degradation:+.0%}) suggests overfitting",
            })
        elif avg_degradation < -0.25:
            flags.append({
                "flag": "MODERATE_OOS_DEGRADATION",
                "severity": "medium",
                "detail": f"Avg Sharpe degradation ({avg_degradation:+.0%}) — monitor",
            })

        # Flag: poor consistency
        if consistency < 0.5:
            flags.append({
                "flag": "POOR_OOS_CONSISTENCY",
                "severity": "high",
                "detail": f"Only {consistency:.0%} of windows have positive test Sharpe",
            })

    # ── 2. Regime collapse ───────────────────────────────────────────────
    if regime_results is not None and not regime_results.empty:
        regime_sharpes = regime_results.set_index("regime")["sharpe"]
        scores["regime_sharpe_range"] = regime_sharpes.max() - regime_sharpes.min()
        scores["regime_sharpe_std"] = regime_sharpes.std()

        # Check if negative Sharpe in any major regime
        major_regimes = regime_results[regime_results["pct_of_total"] > 15]
        negative_regimes = major_regimes[major_regimes["sharpe"] < 0]

        if not negative_regimes.empty:
            names = ", ".join(negative_regimes["regime"].tolist())
            flags.append({
                "flag": "REGIME_COLLAPSE",
                "severity": "high",
                "detail": f"Negative Sharpe in major regime(s): {names}",
            })

        # Check for regime dependence (one regime dominates returns)
        if regime_sharpes.max() > 0 and regime_sharpes.min() < -0.5:
            flags.append({
                "flag": "REGIME_DEPENDENT",
                "severity": "medium",
                "detail": "Strategy performance is highly regime-dependent",
            })

    # ── 3. Parameter instability ─────────────────────────────────────────
    if sensitivity_results is not None and not sensitivity_results.empty:
        sharpe_cv = sensitivity_results["sharpe"].std() / abs(sensitivity_results["sharpe"].mean()) \
            if sensitivity_results["sharpe"].mean() != 0 else float("inf")
        scores["param_stability_cv"] = sharpe_cv

        if sharpe_cv > 1.0:
            flags.append({
                "flag": "UNSTABLE_PARAMETERS",
                "severity": "high",
                "detail": f"Sharpe CV across params ({sharpe_cv:.2f}) shows instability",
            })
        elif sharpe_cv > 0.5:
            flags.append({
                "flag": "MODERATE_PARAM_SENSITIVITY",
                "severity": "medium",
                "detail": f"Sharpe CV ({sharpe_cv:.2f}) — some parameter sensitivity",
            })

        # Check if any params produce negative Sharpe
        negative_pct = (sensitivity_results["sharpe"] < 0).mean()
        if negative_pct > 0.3:
            flags.append({
                "flag": "FRAGILE_PARAMETERS",
                "severity": "medium",
                "detail": f"{negative_pct:.0%} of parameter combos produce negative Sharpe",
            })

    # ── 4. Overall assessment ────────────────────────────────────────────
    high_flags = sum(1 for f in flags if f["severity"] == "high")
    medium_flags = sum(1 for f in flags if f["severity"] == "medium")

    if high_flags >= 2:
        assessment = "LIKELY_OVERFIT"
    elif high_flags == 1:
        assessment = "POSSIBLE_OVERFIT"
    elif medium_flags >= 2:
        assessment = "MONITOR"
    else:
        assessment = "ACCEPTABLE"

    result = {
        "assessment": assessment,
        "flags": flags,
        "scores": scores,
        "n_high_flags": high_flags,
        "n_medium_flags": medium_flags,
    }

    _log_overfitting_report(result)
    return result


def calculate_strategy_stability(
    walkforward_results: pd.DataFrame,
) -> dict:
    """
    Compute stability metrics from walk-forward results.

    Returns
    -------
    dict
        stability_score (0–100), consistency, sharpe metrics.
    """
    if walkforward_results.empty:
        return {"stability_score": 0.0}

    test_sharpes = walkforward_results["test_sharpe"]
    consistency = (test_sharpes > 0).mean()
    mean_sharpe = test_sharpes.mean()
    sharpe_std = test_sharpes.std()
    sharpe_ir = mean_sharpe / sharpe_std if sharpe_std > 0 else 0.0

    # Stability score (0–100)
    score = 0.0
    score += min(consistency * 40, 40)  # Up to 40 pts for consistency
    score += min(max(mean_sharpe, 0) * 20, 30)  # Up to 30 pts for mean Sharpe
    score += min(max(sharpe_ir, 0) * 15, 30)  # Up to 30 pts for IR

    return {
        "stability_score": round(score, 1),
        "consistency": consistency,
        "mean_test_sharpe": mean_sharpe,
        "test_sharpe_std": sharpe_std,
        "sharpe_ir": sharpe_ir,
        "n_windows": len(walkforward_results),
    }


def _log_overfitting_report(result: dict) -> None:
    """Log overfitting detection results."""
    logger.info("\n" + "=" * 60)
    logger.info("OVERFITTING DETECTION REPORT")
    logger.info("=" * 60)
    logger.info(f"  Assessment: {result['assessment']}")
    logger.info(f"  High flags: {result['n_high_flags']}")
    logger.info(f"  Medium flags: {result['n_medium_flags']}")

    for flag in result["flags"]:
        icon = "🔴" if flag["severity"] == "high" else "🟡"
        logger.info(f"  {icon} [{flag['flag']}] {flag['detail']}")

    if not result["flags"]:
        logger.info("  ✓ No overfitting flags detected")
