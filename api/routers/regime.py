"""Regime endpoints — current market regime and behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from contracts import RegimeState

router = APIRouter()


@router.get("/current")
async def get_current_regime() -> dict:
    """Get the current detected market regime and portfolio behavior."""
    # Try new regime intelligence engine first
    try:
        from regimes import get_current_regime as _get_regime, get_regime_behavior
        regime_name, behavior = _get_regime()
        return {
            "regime": regime_name,
            "behavior": {
                "max_equity_weight": behavior.max_equity_weight,
                "covariance_method": behavior.covariance_method,
                "rebalance_drift_threshold": behavior.rebalance_drift_threshold,
                "tilt_strength": behavior.tilt_strength,
            },
        }
    except Exception:
        pass

    # Fallback to legacy regime data
    regime_path = Path("data/processed/regime_analysis.parquet")
    if not regime_path.exists():
        raise HTTPException(status_code=503, detail="Regime data not available. Run the pipeline first.")

    try:
        df = pd.read_parquet(regime_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read regime data: {exc}")

    if df.empty:
        raise HTTPException(status_code=503, detail="Regime data is empty.")

    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").iloc[-1]

    regime_raw = str(latest.get("regime", "unknown")).lower()
    try:
        regime = RegimeState(regime_raw)
    except ValueError:
        regime = RegimeState.SIDEWAYS

    return {
        "as_of": latest["date"].date().isoformat(),
        "regime": regime.value,
        "volatility_state": str(latest.get("volatility_state", "normal")),
        "confidence": float(latest.get("confidence", 0.0)) if "confidence" in latest else None,
    }
