"""Rebalance endpoints — proposed trades."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from contracts import RebalanceDecision, RebalanceMethod, RebalanceProposal, RebalanceTrade
from warehouse import get_warehouse

router = APIRouter()


@router.get("/proposed", response_model=RebalanceProposal)
async def get_proposed_rebalance() -> RebalanceProposal:
    """Get the most recent rebalance proposal."""
    wh = get_warehouse()

    # Load rebalance trades
    try:
        trades_df = wh.read_table("rebalance_trades")
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=503, detail="Rebalance data not available. Run the pipeline first.")

    if trades_df.empty:
        return RebalanceProposal(
            decision=RebalanceDecision(
                should_rebalance=False,
                method=RebalanceMethod.THRESHOLD,
                max_drift=0.0,
                threshold=0.05,
                reason="No rebalance needed — within drift tolerance",
            ),
            trades=[],
        )

    trades: list[RebalanceTrade] = []
    for _, row in trades_df.iterrows():
        delta = float(row.get("target_weight", 0)) - float(row.get("current_weight", 0))
        trades.append(
            RebalanceTrade(
                ticker=str(row["ticker"]),
                direction="BUY" if delta > 0 else "SELL",
                current_weight=float(row.get("current_weight", 0)),
                target_weight=float(row.get("target_weight", 0)),
                delta_weight=delta,
                shares=float(row.get("shares", 0)) if "shares" in row else None,
                estimated_value=float(row.get("estimated_value", 0)) if "estimated_value" in row else None,
            )
        )

    max_drift = max(abs(t.delta_weight) for t in trades) if trades else 0.0

    return RebalanceProposal(
        decision=RebalanceDecision(
            should_rebalance=True,
            method=RebalanceMethod.THRESHOLD,
            max_drift=max_drift,
            threshold=0.05,
            reason=f"Max drift {max_drift:.1%} exceeds threshold",
        ),
        trades=trades,
        estimated_cost=None,
    )
