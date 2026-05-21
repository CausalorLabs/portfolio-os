"""
Monitoring — drift detection, alerts, allocation breach checks.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from contracts import AlertLevel, DriftAlert, DrawdownAlert


def check_drift(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    threshold: float = 0.05,
) -> list[DriftAlert]:
    """Check all positions for drift beyond threshold."""
    alerts: list[DriftAlert] = []
    now = datetime.now(timezone.utc)

    for ticker in set(current_weights) | set(target_weights):
        current = current_weights.get(ticker, 0.0)
        target = target_weights.get(ticker, 0.0)
        drift = abs(current - target)

        if drift > threshold:
            level = AlertLevel.CRITICAL if drift > threshold * 2 else AlertLevel.WARNING
            alerts.append(
                DriftAlert(
                    timestamp=now,
                    level=level,
                    ticker=ticker,
                    current_weight=current,
                    target_weight=target,
                    drift=drift,
                    message=f"{ticker} drifted {drift:.1%} (current={current:.1%}, target={target:.1%})",
                )
            )

    if alerts:
        logger.warning(f"Drift alerts: {len(alerts)} positions exceed {threshold:.1%}")
    return alerts


def check_drawdown(
    nav_series: pd.Series,
    threshold: float = -0.15,
) -> DrawdownAlert | None:
    """Check if current drawdown breaches threshold."""
    if nav_series.empty:
        return None

    peak = nav_series.cummax()
    drawdown = (nav_series - peak) / peak
    current_dd = drawdown.iloc[-1]

    if current_dd < threshold:
        return DrawdownAlert(
            timestamp=datetime.now(timezone.utc),
            level=AlertLevel.CRITICAL if current_dd < threshold * 1.5 else AlertLevel.WARNING,
            current_drawdown=float(current_dd),
            threshold=threshold,
            message=f"Drawdown at {current_dd:.1%} breaches {threshold:.1%} threshold",
        )
    return None


def check_concentration(
    weights: dict[str, float],
    hhi_threshold: float = 0.25,
) -> DriftAlert | None:
    """Check if portfolio concentration (HHI) is too high."""
    hhi = sum(w ** 2 for w in weights.values())
    if hhi > hhi_threshold:
        top_ticker = max(weights, key=weights.get)  # type: ignore[arg-type]
        return DriftAlert(
            timestamp=datetime.now(timezone.utc),
            level=AlertLevel.WARNING,
            ticker=top_ticker,
            current_weight=weights[top_ticker],
            target_weight=0.0,
            drift=hhi,
            message=f"Portfolio HHI={hhi:.3f} exceeds {hhi_threshold:.3f}. Largest position: {top_ticker} at {weights[top_ticker]:.1%}",
        )
    return None
