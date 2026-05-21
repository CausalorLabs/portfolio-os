"""
Regime transitions — transition matrix, duration analysis, stability metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_transition_matrix(regime_series: pd.Series) -> pd.DataFrame:
    """
    Compute the empirical regime transition probability matrix.

    Returns a DataFrame where cell (i, j) = P(regime_t+1 = j | regime_t = i).
    """
    states = sorted(regime_series.dropna().unique())
    n = len(states)
    state_idx = {s: i for i, s in enumerate(states)}

    counts = np.zeros((n, n), dtype=int)
    prev = None
    for regime in regime_series:
        if prev is not None and regime in state_idx and prev in state_idx:
            counts[state_idx[prev]][state_idx[regime]] += 1
        prev = regime

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    probs = counts / row_sums

    return pd.DataFrame(probs, index=states, columns=states)


def compute_regime_durations(regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the duration of each regime episode.

    Returns DataFrame with: regime, start_date, end_date, duration_days
    """
    if regimes.empty:
        return pd.DataFrame(columns=["regime", "start_date", "end_date", "duration_days"])

    episodes = []
    current_regime = regimes.iloc[0]["regime"]
    start_date = regimes.iloc[0]["date"]

    for i in range(1, len(regimes)):
        row = regimes.iloc[i]
        if row["regime"] != current_regime:
            episodes.append({
                "regime": current_regime,
                "start_date": start_date,
                "end_date": regimes.iloc[i - 1]["date"],
                "duration_days": i - episodes[-1].get("_idx", 0) if episodes else i,
            })
            episodes[-1]["_idx"] = i
            current_regime = row["regime"]
            start_date = row["date"]

    # Last episode
    episodes.append({
        "regime": current_regime,
        "start_date": start_date,
        "end_date": regimes.iloc[-1]["date"],
    })

    # Recompute durations from dates
    result = pd.DataFrame(episodes)
    if "_idx" in result.columns:
        result = result.drop(columns=["_idx"])
    result["start_date"] = pd.to_datetime(result["start_date"])
    result["end_date"] = pd.to_datetime(result["end_date"])
    result["duration_days"] = (result["end_date"] - result["start_date"]).dt.days + 1

    return result


def compute_stability_metrics(regimes: pd.DataFrame) -> dict:
    """
    Compute regime stability metrics.

    Returns:
        transitions_per_year: avg regime changes per year
        avg_duration_days: average episode duration
        median_duration_days: median episode duration
        longest_regime: (regime, duration_days)
        shortest_regime: (regime, duration_days)
        dominant_regime: regime with most total days
    """
    if regimes.empty:
        return {}

    durations = compute_regime_durations(regimes)
    if durations.empty:
        return {}

    total_days = (
        pd.to_datetime(regimes["date"].iloc[-1]) - pd.to_datetime(regimes["date"].iloc[0])
    ).days or 1
    total_years = total_days / 365.25
    n_transitions = len(durations) - 1

    # Per-regime stats
    regime_total_days = durations.groupby("regime")["duration_days"].sum()
    dominant = regime_total_days.idxmax()

    longest_idx = durations["duration_days"].idxmax()
    shortest_idx = durations["duration_days"].idxmin()

    return {
        "transitions_per_year": round(n_transitions / total_years, 1) if total_years > 0 else 0,
        "avg_duration_days": round(durations["duration_days"].mean(), 1),
        "median_duration_days": round(durations["duration_days"].median(), 1),
        "longest_regime": (durations.loc[longest_idx, "regime"], int(durations.loc[longest_idx, "duration_days"])),
        "shortest_regime": (durations.loc[shortest_idx, "regime"], int(durations.loc[shortest_idx, "duration_days"])),
        "dominant_regime": dominant,
        "dominant_pct": round(regime_total_days[dominant] / regime_total_days.sum() * 100, 1),
        "n_episodes": len(durations),
    }
