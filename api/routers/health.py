"""Health endpoint."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from contracts import HealthStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """System health check."""
    # Check last ingestion time from processed data
    last_ingestion = None
    inr_path = Path("data/processed/inr_prices.parquet")
    if inr_path.exists():
        mtime = inr_path.stat().st_mtime
        last_ingestion = datetime.fromtimestamp(mtime, tz=timezone.utc)

    return HealthStatus(
        status="ok",
        version="1.0.0-mvp",
        environment="dev",
        last_ingestion=last_ingestion,
    )
