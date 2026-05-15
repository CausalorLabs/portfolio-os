"""
Research scoring system — single quality score for strategy evaluation.

Components (weighted):
  - Sharpe stability     (25%)
  - Drawdown consistency (20%)
  - Turnover efficiency  (20%)
  - Regime robustness    (20%)
  - Parameter robustness (15%)

Output: Research Quality Score 0–100.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def calculate_research_score(
    walkforward_results: pd.DataFrame | None = None,
    regime_results: pd.DataFrame | None = None,
    sensitivity_results: pd.DataFrame | None = None,
    stress_results: pd.DataFrame | None = None,
    signal_decay: pd.DataFrame | None = None,
) -> dict:
    """
    Compute a single research quality score from validation results.

    Returns
    -------
    dict
        total_score (0-100), component scores, grade (A/B/C/D/F).
    """
    components = {}

    # ── 1. Sharpe stability (25%) ────────────────────────────────────────
    sharpe_score = _score_sharpe_stability(walkforward_results)
    components["sharpe_stability"] = {"score": sharpe_score, "weight": 0.25}

    # ── 2. Drawdown consistency (20%) ────────────────────────────────────
    dd_score = _score_drawdown_consistency(walkforward_results)
    components["drawdown_consistency"] = {"score": dd_score, "weight": 0.20}

    # ── 3. Turnover efficiency (20%) ─────────────────────────────────────
    turnover_score = _score_turnover_efficiency(walkforward_results, stress_results)
    components["turnover_efficiency"] = {"score": turnover_score, "weight": 0.20}

    # ── 4. Regime robustness (20%) ───────────────────────────────────────
    regime_score = _score_regime_robustness(regime_results)
    components["regime_robustness"] = {"score": regime_score, "weight": 0.20}

    # ── 5. Parameter robustness (15%) ────────────────────────────────────
    param_score = _score_param_robustness(sensitivity_results)
    components["parameter_robustness"] = {"score": param_score, "weight": 0.15}

    # ── Total ────────────────────────────────────────────────────────────
    total = sum(c["score"] * c["weight"] for c in components.values())
    total = round(min(total, 100), 1)

    # Grade
    if total >= 80:
        grade = "A"
    elif total >= 65:
        grade = "B"
    elif total >= 50:
        grade = "C"
    elif total >= 35:
        grade = "D"
    else:
        grade = "F"

    result = {
        "total_score": total,
        "grade": grade,
        "components": {k: v["score"] for k, v in components.items()},
        "weights": {k: v["weight"] for k, v in components.items()},
    }

    _log_score(result)
    return result


def _score_sharpe_stability(wf: pd.DataFrame | None) -> float:
    """Score 0–100 for Sharpe stability across walk-forward windows."""
    if wf is None or wf.empty:
        return 50.0  # Neutral if no data

    test_sharpes = wf["test_sharpe"]
    consistency = (test_sharpes > 0).mean()
    avg_sharpe = max(test_sharpes.mean(), 0)
    sharpe_std = test_sharpes.std()
    ir = avg_sharpe / sharpe_std if sharpe_std > 0 else 0

    score = 0.0
    score += min(consistency * 50, 50)  # Up to 50 for consistency
    score += min(avg_sharpe * 25, 30)   # Up to 30 for mean Sharpe
    score += min(max(ir, 0) * 10, 20)   # Up to 20 for IR

    return min(round(score, 1), 100)


def _score_drawdown_consistency(wf: pd.DataFrame | None) -> float:
    """Score 0–100 for drawdown behavior across windows."""
    if wf is None or wf.empty:
        return 50.0

    test_dd = wf["test_max_drawdown"].abs()
    avg_dd = test_dd.mean()
    max_dd = test_dd.max()

    score = 100.0
    # Penalize for large average drawdown
    if avg_dd > 0.30:
        score -= 40
    elif avg_dd > 0.20:
        score -= 25
    elif avg_dd > 0.10:
        score -= 10

    # Penalize for extreme worst-case
    if max_dd > 0.50:
        score -= 30
    elif max_dd > 0.35:
        score -= 15

    return max(round(score, 1), 0)


def _score_turnover_efficiency(
    wf: pd.DataFrame | None,
    stress: pd.DataFrame | None,
) -> float:
    """Score 0–100 for turnover and friction efficiency."""
    score = 70.0  # Base — adjusted by friction evidence

    if stress is not None and not stress.empty:
        avg_cagr_impact = stress["cagr_impact"].mean()
        if avg_cagr_impact > -0.02:
            score += 20
        elif avg_cagr_impact > -0.05:
            score += 10
        else:
            score -= 15

    if wf is not None and not wf.empty:
        # Consistent test CAGRs are a sign of controlled friction
        test_cagr_std = wf["test_cagr"].std()
        if test_cagr_std < 0.05:
            score += 10

    return min(round(score, 1), 100)


def _score_regime_robustness(reg: pd.DataFrame | None) -> float:
    """Score 0–100 for regime consistency."""
    if reg is None or reg.empty:
        return 50.0

    positive_regimes = (reg["sharpe"] > 0).sum()
    total = len(reg)
    pct_positive = positive_regimes / total

    worst_sharpe = reg["sharpe"].min()

    score = pct_positive * 70  # Up to 70 for positive regimes
    if worst_sharpe > -0.5:
        score += 20
    elif worst_sharpe > -1.0:
        score += 10

    # Bonus for balanced performance
    sharpe_std = reg["sharpe"].std()
    if sharpe_std < 0.5:
        score += 10

    return min(round(score, 1), 100)


def _score_param_robustness(sens: pd.DataFrame | None) -> float:
    """Score 0–100 for parameter sensitivity."""
    if sens is None or sens.empty:
        return 50.0

    sharpe_mean = sens["sharpe"].mean()
    sharpe_std = sens["sharpe"].std()
    cv = sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else float("inf")
    positive_pct = (sens["sharpe"] > 0).mean()

    score = 0.0
    # Low CV = stable
    if cv < 0.2:
        score += 50
    elif cv < 0.5:
        score += 35
    elif cv < 1.0:
        score += 20
    else:
        score += 5

    # High positive percentage
    score += positive_pct * 50

    return min(round(score, 1), 100)


def _log_score(result: dict) -> None:
    """Log research score."""
    logger.info("\n" + "=" * 60)
    logger.info("RESEARCH QUALITY SCORE")
    logger.info("=" * 60)
    logger.info(f"  Total Score:  {result['total_score']}/100  (Grade: {result['grade']})")
    logger.info("")

    for name, score in result["components"].items():
        weight = result["weights"][name]
        logger.info(f"  {name:25s}: {score:>5.1f}/100  (×{weight:.0%})")
