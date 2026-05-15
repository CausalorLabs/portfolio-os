"""
Rebalance logic — determines WHEN rebalancing should happen.

Supports calendar-based, threshold-based, and hybrid triggers.
"""

import pandas as pd
from loguru import logger


def should_rebalance(
    current_weights: pd.DataFrame,
    target_weights: pd.DataFrame,
    drift_threshold: float = 0.05,
    method: str = "threshold",
    last_rebalance_date: pd.Timestamp | None = None,
    current_date: pd.Timestamp | None = None,
    calendar_freq: str = "quarterly",
) -> dict:
    """
    Determine whether a rebalance should be triggered.

    Parameters
    ----------
    current_weights : pd.DataFrame
        Columns: ticker, current_weight.
    target_weights : pd.DataFrame
        Columns: ticker, target_weight.
    drift_threshold : float
        Max tolerable drift for any single asset (e.g. 0.05 = 5%).
    method : str
        "threshold", "calendar", or "hybrid".
    last_rebalance_date : pd.Timestamp, optional
        Date of last rebalance (for calendar method).
    current_date : pd.Timestamp, optional
        Current date (for calendar method).
    calendar_freq : str
        "monthly" or "quarterly".

    Returns
    -------
    dict
        Keys: should_rebalance (bool), reason (str), max_drift (float),
              method (str), assets_drifted (list)
    """
    merged = target_weights[["ticker", "target_weight"]].merge(
        current_weights[["ticker", "current_weight"]],
        on="ticker",
        how="outer",
    )
    merged["current_weight"] = merged["current_weight"].fillna(0.0)
    merged["target_weight"] = merged["target_weight"].fillna(0.0)
    merged["drift"] = (merged["current_weight"] - merged["target_weight"]).abs()
    max_drift = merged["drift"].max()
    drifted_assets = merged[merged["drift"] > drift_threshold]["ticker"].tolist()

    # Threshold check
    threshold_triggered = max_drift > drift_threshold

    # Calendar check
    calendar_triggered = False
    if last_rebalance_date is not None and current_date is not None:
        days_elapsed = (current_date - last_rebalance_date).days
        if calendar_freq == "monthly":
            calendar_triggered = days_elapsed >= 28
        elif calendar_freq == "quarterly":
            calendar_triggered = days_elapsed >= 84
        else:
            calendar_triggered = days_elapsed >= 84

    if method == "threshold":
        trigger = threshold_triggered
        reason = f"drift {max_drift:.2%} > {drift_threshold:.0%}" if trigger else "within threshold"
    elif method == "calendar":
        trigger = calendar_triggered
        reason = f"calendar ({calendar_freq})" if trigger else "not due yet"
    elif method == "hybrid":
        trigger = threshold_triggered or calendar_triggered
        reasons = []
        if threshold_triggered:
            reasons.append(f"drift {max_drift:.2%}")
        if calendar_triggered:
            reasons.append(f"calendar ({calendar_freq})")
        reason = " + ".join(reasons) if reasons else "within threshold & not due"
    else:
        trigger = threshold_triggered
        reason = f"unknown method '{method}', used threshold"

    result = {
        "should_rebalance": trigger,
        "reason": reason,
        "max_drift": max_drift,
        "method": method,
        "assets_drifted": drifted_assets,
    }

    status = "YES" if trigger else "NO"
    logger.info(f"Rebalance check ({method}): {status} — {reason}")

    return result


def calculate_rebalance_trades(
    current_weights: pd.DataFrame,
    target_weights: pd.DataFrame,
    portfolio_value: float,
) -> pd.DataFrame:
    """
    Calculate the trades needed to rebalance from current to target.

    Parameters
    ----------
    current_weights : pd.DataFrame
        Columns: ticker, current_weight.
    target_weights : pd.DataFrame
        Columns: ticker, target_weight.
    portfolio_value : float
        Total portfolio value in INR.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, current_weight, target_weight, weight_change,
                 trade_value_inr, action
    """
    df = target_weights[["ticker", "target_weight"]].merge(
        current_weights[["ticker", "current_weight"]],
        on="ticker",
        how="outer",
    )
    df["current_weight"] = df["current_weight"].fillna(0.0)
    df["target_weight"] = df["target_weight"].fillna(0.0)

    df["weight_change"] = df["target_weight"] - df["current_weight"]
    df["trade_value_inr"] = df["weight_change"] * portfolio_value
    df["action"] = df["weight_change"].apply(
        lambda x: "BUY" if x > 0.001 else ("SELL" if x < -0.001 else "HOLD")
    )

    trades = df[df["action"] != "HOLD"].copy()
    if not trades.empty:
        logger.info(f"Rebalance trades ({len(trades)}):")
        for _, row in trades.iterrows():
            logger.info(
                f"  {row['action']:4s} {row['ticker']:15s}  "
                f"₹{abs(row['trade_value_inr']):>12,.0f}"
            )

    return df
