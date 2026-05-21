"""
Portfolio OS — FastAPI service.

Endpoints:
  /health              — system health check
  /portfolio/current   — current portfolio state
  /portfolio/history   — historical NAV + returns
  /rebalance/proposed  — proposed rebalance trades
  /regime/current      — current market regime
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from loguru import logger

from api.routers import health, portfolio, rebalance, regime
from warehouse import get_warehouse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown: register warehouse tables."""
    wh = get_warehouse()
    registered = wh.register_all()
    logger.info(f"API startup: {len(registered)} warehouse tables available")
    yield
    wh.close()
    logger.info("API shutdown: warehouse closed")


app = FastAPI(
    title="Portfolio OS",
    description="Intelligent Portfolio Operating System — Decision Support API",
    version="1.0.0-mvp",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
app.include_router(rebalance.router, prefix="/rebalance", tags=["rebalance"])
app.include_router(regime.router, prefix="/regime", tags=["regime"])
