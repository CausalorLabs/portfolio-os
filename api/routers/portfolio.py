"""Portfolio endpoints — current state, history, metrics."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from contracts import (
    AllocationWeight,
    PortfolioMetrics,
    PortfolioNAV,
    PortfolioSummary,
)
from warehouse import get_warehouse

router = APIRouter()


@router.get("/current", response_model=PortfolioSummary)
async def get_current_portfolio() -> PortfolioSummary:
    """Get current portfolio state: NAV, metrics, holdings."""
    wh = get_warehouse()

    # Load NAV
    try:
        nav_df = wh.read_table("portfolio_nav")
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=503, detail="Portfolio NAV data not available. Run the pipeline first.")

    if nav_df.empty:
        raise HTTPException(status_code=503, detail="Portfolio NAV is empty.")

    latest = nav_df.iloc[-1]
    nav_value = float(latest.get("portfolio_nav", 0))
    as_of = pd.to_datetime(latest.get("date", date.today())).date()

    # Load metrics (long-format CSV: metric, value)
    metrics_path = Path("reports/portfolio_metrics.csv")
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        if not mdf.empty:
            m = dict(zip(mdf["metric"], mdf["value"]))
            metrics = PortfolioMetrics(
                cagr=float(m.get("cagr", 0)),
                sharpe_ratio=float(m.get("sharpe_ratio", 0)),
                sortino_ratio=float(m.get("sortino_ratio", 0)) if "sortino_ratio" in m else None,
                max_drawdown=float(m.get("max_drawdown", 0)),
                calmar_ratio=float(m.get("calmar_ratio", 0)) if "calmar_ratio" in m else None,
                annualized_volatility=float(m.get("annualized_volatility", 0)),
                portfolio_nav=nav_value,
            )
        else:
            raise HTTPException(status_code=503, detail="Metrics CSV empty.")
    else:
        raise HTTPException(status_code=503, detail="Metrics not computed. Run the pipeline first.")

    # Load target weights
    holdings: list[AllocationWeight] = []
    try:
        weights_df = wh.read_table("target_weights")
        if not weights_df.empty:
            for _, row in weights_df.iterrows():
                holdings.append(
                    AllocationWeight(
                        ticker=str(row["ticker"]),
                        weight=float(row.get("target_weight", row.get("weight", 0))),
                        method=str(row.get("strategy", row.get("method", "hrp"))),
                    )
                )
    except (FileNotFoundError, ValueError):
        pass

    return PortfolioSummary(
        as_of=as_of,
        nav=nav_value,
        metrics=metrics,
        holdings=holdings,
    )


@router.get("/history", response_model=list[PortfolioNAV])
async def get_portfolio_history(
    start: date | None = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end: date | None = Query(default=None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(default=252, ge=1, le=5000, description="Max rows to return"),
) -> list[PortfolioNAV]:
    """Get historical portfolio NAV."""
    wh = get_warehouse()

    try:
        nav_df = wh.read_table("portfolio_nav")
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=503, detail="NAV data not available.")

    nav_df["date"] = pd.to_datetime(nav_df["date"])

    if start:
        nav_df = nav_df[nav_df["date"] >= pd.Timestamp(start)]
    if end:
        nav_df = nav_df[nav_df["date"] <= pd.Timestamp(end)]

    nav_df = nav_df.tail(limit)

    return [
        PortfolioNAV(
            date=row["date"].date(),
            portfolio_nav=float(row["portfolio_nav"]),
            daily_return=float(row["daily_return"]) if "daily_return" in row and pd.notna(row.get("daily_return")) else None,
        )
        for _, row in nav_df.iterrows()
    ]
