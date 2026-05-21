"""
ML Alpha Engine — Model Evaluation Framework.

Evaluates models by PORTFOLIO UTILITY, not prediction accuracy.

Metrics:
  - IC (information coefficient — ranking quality)
  - Hit ratio (% correct direction)
  - Turnover impact (operational realism)
  - Drawdown improvement (risk value)
  - Sharpe contribution (portfolio utility)
  - Stability (IC consistency)
  - Calibration (confidence vs accuracy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr


# ── Core metrics ─────────────────────────────────────────────────────────────


def compute_rank_ic(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Information Coefficient — Spearman rank correlation.

    The PRIMARY metric for cross-sectional ranking models.
    IC > 0.05 is considered meaningful in practice.
    """
    valid = ~(predictions.isna() | actuals.isna())
    if valid.sum() < 3:
        return 0.0
    ic, _ = spearmanr(predictions[valid], actuals[valid])
    return float(ic) if not np.isnan(ic) else 0.0


def compute_daily_ic(
    predictions: pd.DataFrame,
    actuals_col: str = "actual",
    pred_col: str = "prediction",
) -> pd.DataFrame:
    """
    Compute IC per day (cross-sectional Spearman).

    Returns: date | ic
    """
    records = []
    for dt, grp in predictions.groupby("date"):
        if len(grp) < 3:
            continue
        ic = compute_rank_ic(grp[pred_col], grp[actuals_col])
        records.append({"date": dt, "ic": ic})

    return pd.DataFrame(records)


def compute_hit_ratio(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Proportion of correct directional predictions.

    For ranking targets: hit = predicted rank > 0.5 when actual > 0.5.
    """
    valid = ~(predictions.isna() | actuals.isna())
    if valid.sum() < 3:
        return 0.5

    pred_above = predictions[valid] > 0.5
    actual_above = actuals[valid] > 0.5
    return float((pred_above == actual_above).mean())


def compute_turnover_impact(
    alpha_scores: pd.DataFrame,
    base_weights: pd.DataFrame | None = None,
    tilt_strength: float = 0.20,
) -> float:
    """
    Estimate excess turnover caused by alpha tilts.

    Returns annualized excess turnover as fraction.
    """
    if alpha_scores.empty:
        return 0.0

    # Compute rank changes over time
    alpha = alpha_scores.sort_values(["ticker", "date"]).copy()
    alpha["rank_change"] = alpha.groupby("ticker")["rank"].diff().abs()

    # Average daily rank change × tilt strength ≈ daily excess turnover
    avg_daily_change = alpha["rank_change"].mean()
    if np.isnan(avg_daily_change):
        return 0.0

    # Annualize (252 trading days)
    annual_turnover = avg_daily_change * tilt_strength * 252
    return float(annual_turnover)


def compute_drawdown_improvement(
    alpha_nav: pd.Series,
    baseline_nav: pd.Series,
) -> float:
    """
    Compare max drawdown: alpha-tilted portfolio vs baseline.

    Returns improvement as positive fraction (0.1 = 10% better DD).
    """
    def _max_dd(nav):
        peak = nav.cummax()
        dd = (nav - peak) / peak
        return dd.min()

    if len(alpha_nav) < 10 or len(baseline_nav) < 10:
        return 0.0

    dd_alpha = _max_dd(alpha_nav)
    dd_baseline = _max_dd(baseline_nav)

    if dd_baseline == 0:
        return 0.0

    # Positive = alpha has shallower drawdown
    improvement = (dd_alpha - dd_baseline) / abs(dd_baseline)
    return float(improvement)


def compute_sharpe_contribution(
    alpha_returns: pd.Series,
    baseline_returns: pd.Series,
    risk_free_annual: float = 0.065,
) -> float:
    """
    Marginal Sharpe ratio contribution from alpha tilts.

    Returns difference in annualized Sharpe.
    """
    rf_daily = risk_free_annual / 252

    def _sharpe(rets):
        excess = rets - rf_daily
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(252))

    return _sharpe(alpha_returns) - _sharpe(baseline_returns)


def compute_ic_stability(daily_ic: pd.DataFrame) -> float:
    """
    Stability of IC over time (lower std = more stable).

    Returns IC_mean / IC_std (like Sharpe of IC).
    """
    if daily_ic.empty or "ic" not in daily_ic.columns:
        return 0.0

    mean_ic = daily_ic["ic"].mean()
    std_ic = daily_ic["ic"].std()

    if std_ic == 0:
        return 0.0

    return float(mean_ic / std_ic)


def compute_calibration(
    confidence: pd.Series,
    accuracy: pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Confidence calibration: does 80% confidence mean 80% accuracy?

    Returns: confidence_bin | mean_confidence | mean_accuracy | count
    """
    valid = ~(confidence.isna() | accuracy.isna())
    if valid.sum() < 10:
        return pd.DataFrame()

    df = pd.DataFrame({
        "confidence": confidence[valid],
        "accuracy": accuracy[valid],
    })

    df["bin"] = pd.qcut(df["confidence"], n_bins, duplicates="drop")
    cal = df.groupby("bin", observed=True).agg(
        mean_confidence=("confidence", "mean"),
        mean_accuracy=("accuracy", "mean"),
        count=("confidence", "count"),
    ).reset_index()

    return cal


# ── Full evaluation report ──────────────────────────────────────────────────


def evaluate_alpha_model(
    predictions: pd.DataFrame,
    alpha_scores: pd.DataFrame | None = None,
    baseline_nav: pd.Series | None = None,
    alpha_nav: pd.Series | None = None,
) -> dict:
    """
    Comprehensive model evaluation.

    Parameters
    ----------
    predictions : DataFrame with date, ticker, prediction, actual
    alpha_scores : Full alpha scores with confidence
    baseline_nav : NAV of baseline portfolio (no alpha tilts)
    alpha_nav : NAV of alpha-tilted portfolio

    Returns
    -------
    dict with all evaluation metrics
    """
    logger.info("Evaluating alpha model...")

    metrics = {}

    # IC
    overall_ic = compute_rank_ic(predictions["prediction"], predictions["actual"])
    metrics["rank_ic"] = round(overall_ic, 4)

    daily_ic = compute_daily_ic(predictions)
    metrics["mean_daily_ic"] = round(daily_ic["ic"].mean(), 4) if not daily_ic.empty else 0
    metrics["ic_stability"] = round(compute_ic_stability(daily_ic), 4)

    # Hit ratio
    metrics["hit_ratio"] = round(compute_hit_ratio(
        predictions["prediction"], predictions["actual"]
    ), 4)

    # Turnover
    if alpha_scores is not None:
        metrics["turnover_impact"] = round(compute_turnover_impact(alpha_scores), 4)

    # Drawdown + Sharpe (if NAV available)
    if baseline_nav is not None and alpha_nav is not None:
        metrics["drawdown_improvement"] = round(
            compute_drawdown_improvement(alpha_nav, baseline_nav), 4
        )
        baseline_rets = baseline_nav.pct_change().dropna()
        alpha_rets = alpha_nav.pct_change().dropna()
        metrics["sharpe_contribution"] = round(
            compute_sharpe_contribution(alpha_rets, baseline_rets), 4
        )

    # Grade
    ic = metrics["rank_ic"]
    if ic >= 0.08:
        metrics["grade"] = "A"
    elif ic >= 0.05:
        metrics["grade"] = "B"
    elif ic >= 0.02:
        metrics["grade"] = "C"
    else:
        metrics["grade"] = "D"

    logger.info(f"  IC={metrics['rank_ic']}, hit={metrics['hit_ratio']}, "
                f"stability={metrics['ic_stability']}, grade={metrics['grade']}")

    return metrics
