"""
Risk Engine — Risk Model Evaluation.

How do we know our risk engine is actually working?

Metrics:
  - Volatility prediction accuracy
  - Drawdown improvement vs baseline
  - Risk budget stability
  - CVaR back-testing
  - Turnover impact
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def evaluate_vol_prediction(
    predicted_vol: pd.Series,
    realized_vol: pd.Series,
) -> dict:
    """
    How well does our EWMA vol predict actual realized vol?

    Returns: MAE, RMSE, bias, rank correlation.
    """
    aligned = pd.DataFrame({
        "predicted": predicted_vol,
        "realized": realized_vol,
    }).dropna()

    if aligned.empty:
        return {"status": "no_data"}

    errors = aligned["predicted"] - aligned["realized"]
    abs_errors = errors.abs()

    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "bias": float(errors.mean()),
        "rank_correlation": float(aligned.corr(method="spearman").iloc[0, 1]),
        "n_observations": len(aligned),
    }


def evaluate_drawdown_improvement(
    risk_nav: pd.Series,
    baseline_nav: pd.Series,
) -> dict:
    """
    Compare drawdown characteristics of risk-managed vs baseline portfolio.
    """
    def _max_dd(nav):
        peak = nav.cummax()
        dd = (nav - peak) / peak
        return float(dd.min())

    def _avg_dd(nav):
        peak = nav.cummax()
        dd = (nav - peak) / peak
        return float(dd.mean())

    risk_dd = _max_dd(risk_nav)
    base_dd = _max_dd(baseline_nav)

    return {
        "risk_managed_max_dd": risk_dd,
        "baseline_max_dd": base_dd,
        "dd_improvement": base_dd - risk_dd,
        "dd_improvement_pct": (base_dd - risk_dd) / abs(base_dd) if base_dd != 0 else 0,
        "risk_managed_avg_dd": _avg_dd(risk_nav),
        "baseline_avg_dd": _avg_dd(baseline_nav),
    }


def evaluate_risk_budget_stability(
    risk_contributions_history: list[dict],
) -> dict:
    """
    How stable are risk contributions over time?

    Low volatility of risk contributions = stable risk model.
    """
    if not risk_contributions_history:
        return {"status": "no_data"}

    df = pd.DataFrame(risk_contributions_history)

    vol_of_contrib = df.std()
    max_change = df.diff().abs().max()

    return {
        "avg_risk_contribution_vol": float(vol_of_contrib.mean()),
        "max_single_period_change": float(max_change.max()),
        "most_volatile_asset": vol_of_contrib.idxmax(),
        "n_periods": len(df),
    }


def evaluate_cvar_backtesting(
    returns: pd.Series,
    predicted_cvar: pd.Series,
    confidence: float = 0.95,
) -> dict:
    """
    CVaR back-test: how often do actual losses exceed predicted CVaR?

    Expected exceedance rate = 1 - confidence = 5%.
    """
    aligned = pd.DataFrame({
        "returns": returns,
        "predicted_cvar": predicted_cvar,
    }).dropna()

    if aligned.empty:
        return {"status": "no_data"}

    exceedances = aligned["returns"] < aligned["predicted_cvar"]
    actual_rate = float(exceedances.mean())
    expected_rate = 1 - confidence

    return {
        "actual_exceedance_rate": actual_rate,
        "expected_exceedance_rate": expected_rate,
        "ratio": actual_rate / expected_rate if expected_rate > 0 else np.nan,
        "n_exceedances": int(exceedances.sum()),
        "n_observations": len(aligned),
        "is_conservative": actual_rate < expected_rate,
    }


def evaluate_turnover_impact(
    weights_history: list[pd.Series],
) -> dict:
    """
    Turnover from risk-based rebalancing.
    """
    if len(weights_history) < 2:
        return {"status": "insufficient_data"}

    turnovers = []
    for i in range(1, len(weights_history)):
        prev = weights_history[i - 1]
        curr = weights_history[i]
        common = prev.index.intersection(curr.index)
        turnover = float((prev[common] - curr[common]).abs().sum()) / 2
        turnovers.append(turnover)

    return {
        "avg_turnover": float(np.mean(turnovers)),
        "max_turnover": float(np.max(turnovers)),
        "min_turnover": float(np.min(turnovers)),
        "total_turnover": float(np.sum(turnovers)),
        "n_rebalances": len(turnovers),
    }


def compute_risk_adjusted_utility(
    returns: pd.Series,
    risk_aversion: float = 2.0,
) -> float:
    """
    Mean-variance utility = E[r] - 0.5 * λ * σ²

    Higher utility = better risk-adjusted performance.
    """
    ann_return = returns.mean() * 252
    ann_var = returns.var() * 252

    return float(ann_return - 0.5 * risk_aversion * ann_var)


def build_risk_evaluation_report(
    risk_nav: pd.Series,
    baseline_nav: pd.Series,
    risk_returns: pd.Series | None = None,
    baseline_returns: pd.Series | None = None,
) -> dict:
    """Full risk model evaluation."""
    report = {}

    report["drawdown"] = evaluate_drawdown_improvement(risk_nav, baseline_nav)

    if risk_returns is not None and baseline_returns is not None:
        report["risk_utility"] = compute_risk_adjusted_utility(risk_returns)
        report["baseline_utility"] = compute_risk_adjusted_utility(baseline_returns)
        report["utility_improvement"] = report["risk_utility"] - report["baseline_utility"]

        # Sharpe comparison
        risk_sharpe = risk_returns.mean() / risk_returns.std() * np.sqrt(252)
        base_sharpe = baseline_returns.mean() / baseline_returns.std() * np.sqrt(252)
        report["risk_sharpe"] = float(risk_sharpe)
        report["baseline_sharpe"] = float(base_sharpe)
        report["sharpe_improvement"] = float(risk_sharpe - base_sharpe)

    return report
