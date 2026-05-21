"""
Regime evaluation framework — validate that regime detection is useful.

Tests:
  1. Stability: are regimes sensible? (not too noisy, not too slow)
  2. Predictive value: do forward returns/vol differ by regime?
  3. Portfolio impact: does regime awareness improve risk-adjusted returns?
  4. Crisis alignment: do panic regimes align with known market crises?
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger


# ── Known crisis periods ─────────────────────────────────────────────────────

KNOWN_CRISES = [
    {"name": "COVID Crash", "start": date(2020, 2, 19), "end": date(2020, 3, 23)},
    {"name": "2022 Rate Shock", "start": date(2022, 1, 3), "end": date(2022, 6, 16)},
    {"name": "SVB/Banking Stress", "start": date(2023, 3, 8), "end": date(2023, 3, 27)},
    {"name": "2024 Yen Carry Unwind", "start": date(2024, 7, 31), "end": date(2024, 8, 12)},
]


def evaluate_predictive_value(
    regimes: pd.DataFrame,
    nav_series: pd.DataFrame,
    forward_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Measure forward returns and volatility per regime.

    For each regime, compute:
      - avg forward 5d, 20d, 60d returns
      - avg forward 20d realized vol
      - prob of negative forward 20d return

    Args:
        regimes: DataFrame with date, regime columns
        nav_series: DataFrame with date, portfolio_nav columns
        forward_windows: Days ahead to measure (default [5, 20, 60])

    Returns:
        DataFrame with one row per regime and forward metrics.
    """
    if forward_windows is None:
        forward_windows = [5, 20, 60]

    regimes = regimes.copy()
    nav = nav_series.copy()

    regimes["date"] = pd.to_datetime(regimes["date"])
    nav["date"] = pd.to_datetime(nav["date"])

    merged = regimes.merge(nav[["date", "portfolio_nav"]], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    if merged.empty:
        return pd.DataFrame()

    # Compute forward returns
    for w in forward_windows:
        merged[f"fwd_return_{w}d"] = merged["portfolio_nav"].pct_change(w).shift(-w)

    # Forward volatility (20d)
    daily_ret = merged["portfolio_nav"].pct_change()
    merged["fwd_vol_20d"] = daily_ret.rolling(20).std().shift(-20) * np.sqrt(252)

    # Prob of loss
    merged["fwd_negative_20d"] = (merged["fwd_return_20d"] < 0).astype(float) if "fwd_return_20d" in merged.columns else np.nan

    # Aggregate by regime
    agg_cols = {f"fwd_return_{w}d": "mean" for w in forward_windows}
    agg_cols["fwd_vol_20d"] = "mean"
    agg_cols["fwd_negative_20d"] = "mean"

    result = merged.groupby("regime").agg(
        n_days=("regime", "count"),
        **{col: pd.NamedAgg(column=col, aggfunc=func) for col, func in agg_cols.items()},
    ).reset_index()

    return result


def evaluate_crisis_alignment(
    regimes: pd.DataFrame,
    crises: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Check if panic/risk_off regimes align with known crisis periods.

    Returns DataFrame with: crisis_name, start, end, pct_panic, pct_risk_off, pct_defensive
    """
    if crises is None:
        crises = KNOWN_CRISES

    regimes = regimes.copy()
    regimes["date"] = pd.to_datetime(regimes["date"])

    results = []
    for crisis in crises:
        start = pd.Timestamp(crisis["start"])
        end = pd.Timestamp(crisis["end"])

        mask = (regimes["date"] >= start) & (regimes["date"] <= end)
        crisis_regimes = regimes[mask]

        if crisis_regimes.empty:
            results.append({
                "crisis": crisis["name"],
                "start": crisis["start"],
                "end": crisis["end"],
                "n_days": 0,
                "pct_panic": None,
                "pct_risk_off": None,
                "pct_defensive": None,
            })
            continue

        counts = crisis_regimes["regime"].value_counts(normalize=True)
        n = len(crisis_regimes)

        results.append({
            "crisis": crisis["name"],
            "start": crisis["start"],
            "end": crisis["end"],
            "n_days": n,
            "pct_panic": round(counts.get("panic", 0) * 100, 1),
            "pct_risk_off": round(counts.get("risk_off", 0) * 100, 1),
            "pct_defensive": round((counts.get("panic", 0) + counts.get("risk_off", 0)) * 100, 1),
        })

    return pd.DataFrame(results)


def evaluate_regime_quality(
    regimes: pd.DataFrame,
    nav_series: pd.DataFrame,
) -> dict:
    """
    Comprehensive regime quality score.

    Criteria:
      1. Stability: 3-15 transitions/year is healthy (score 0-25)
      2. Predictive: panic fwd returns < risk_on fwd returns (score 0-25)
      3. Crisis alignment: defensive % during crises > 50% (score 0-25)
      4. Separation: vol differs meaningfully across regimes (score 0-25)

    Returns dict with component scores and total (0-100).
    """
    from regimes.transitions import compute_stability_metrics

    # 1. Stability
    stability = compute_stability_metrics(regimes)
    tpy = stability.get("transitions_per_year", 0)
    if 3 <= tpy <= 15:
        stability_score = 25
    elif 1 <= tpy < 3 or 15 < tpy <= 25:
        stability_score = 15
    else:
        stability_score = 5

    # 2. Predictive value
    pred = evaluate_predictive_value(regimes, nav_series)
    predictive_score = 5
    if not pred.empty and "fwd_return_20d" in pred.columns:
        regime_returns = dict(zip(pred["regime"], pred["fwd_return_20d"]))
        panic_ret = regime_returns.get("panic", 0)
        risk_on_ret = regime_returns.get("risk_on", 0)
        if panic_ret < risk_on_ret:
            predictive_score = 25
        elif panic_ret < 0:
            predictive_score = 15

    # 3. Crisis alignment
    crisis_df = evaluate_crisis_alignment(regimes)
    crisis_score = 5
    if not crisis_df.empty:
        avg_defensive = crisis_df["pct_defensive"].dropna().mean()
        if avg_defensive > 60:
            crisis_score = 25
        elif avg_defensive > 40:
            crisis_score = 15
        elif avg_defensive > 20:
            crisis_score = 10

    # 4. Vol separation
    separation_score = 5
    if not pred.empty and "fwd_vol_20d" in pred.columns:
        vol_by_regime = dict(zip(pred["regime"], pred["fwd_vol_20d"]))
        panic_vol = vol_by_regime.get("panic", 0)
        risk_on_vol = vol_by_regime.get("risk_on", 0)
        if panic_vol > risk_on_vol * 1.3:
            separation_score = 25
        elif panic_vol > risk_on_vol:
            separation_score = 15

    total = stability_score + predictive_score + crisis_score + separation_score

    grade = "A" if total >= 80 else "B" if total >= 60 else "C" if total >= 40 else "D"

    result = {
        "stability_score": stability_score,
        "predictive_score": predictive_score,
        "crisis_alignment_score": crisis_score,
        "separation_score": separation_score,
        "total_score": total,
        "grade": grade,
        "transitions_per_year": tpy,
    }

    logger.info(
        f"Regime quality: {total}/100 ({grade}) — "
        f"stability={stability_score}, predictive={predictive_score}, "
        f"crisis={crisis_score}, separation={separation_score}"
    )

    return result
