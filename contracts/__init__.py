"""
Data contracts — Pydantic models for all core data tables.

These schemas enforce structure at system boundaries:
  - ingestion output
  - warehouse reads
  - API responses
  - inter-module data handoff
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class AssetType(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    MF = "mf"
    FIXED_INCOME = "fixed_income"
    METAL = "metal"
    FX = "fx"


class Country(str, Enum):
    IN = "IN"
    US = "US"
    GLOBAL = "GLOBAL"


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"


class RebalanceMethod(str, Enum):
    THRESHOLD = "threshold"
    CALENDAR = "calendar"
    HYBRID = "hybrid"


class OptimizationMethod(str, Enum):
    HRP = "hrp"
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOL = "inverse_vol"
    RISK_PARITY = "risk_parity"


class RegimeState(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    SIDEWAYS = "sideways"


# ── Core Data Tables ─────────────────────────────────────────────────────────


class HoldingsSnapshot(BaseModel):
    """Single row in the holdings table."""
    date: date
    ticker: str
    quantity: float = Field(ge=0)
    avg_cost: float = Field(ge=0)
    currency: Currency
    asset_class: AssetType
    sector: str | None = None
    country: Country


class PriceRecord(BaseModel):
    """Single row in the price_history table."""
    date: date
    ticker: str
    close: float
    adj_close: float | None = None
    volume: float | None = Field(default=None, ge=0)
    currency: Currency


class FXRate(BaseModel):
    """Single row in the fx_rates table."""
    date: date
    pair: str  # e.g. "USDINR"
    rate: float = Field(gt=0)


class INRPrice(BaseModel):
    """Normalized price in INR (post-FX conversion)."""
    date: date
    ticker: str
    inr_price: float
    fx_rate: float = Field(gt=0)


# ── Asset Master ─────────────────────────────────────────────────────────────


class AssetConfig(BaseModel):
    """Single row from asset_master.csv."""
    ticker: str
    asset_name: str
    asset_type: AssetType
    country: Country
    currency: Currency
    exchange: str
    annual_rate: float | None = None  # fixed-income only


# ── Portfolio & Analytics ────────────────────────────────────────────────────


class PortfolioNAV(BaseModel):
    """Daily portfolio NAV record."""
    date: date
    portfolio_nav: float
    daily_return: float | None = None


class PortfolioMetrics(BaseModel):
    """Aggregate portfolio risk metrics."""
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float | None = None
    max_drawdown: float
    calmar_ratio: float | None = None
    annualized_volatility: float
    portfolio_nav: float | None = None


class DrawdownPeriod(BaseModel):
    """A single drawdown event."""
    start_date: date
    trough_date: date
    end_date: date | None = None
    depth: float  # negative fraction
    duration_days: int
    recovery_days: int | None = None


# ── Features & Signals ──────────────────────────────────────────────────────


class FeatureRecord(BaseModel):
    """Single feature value in the long-format feature store."""
    date: date
    ticker: str
    feature_name: str
    value: float


class SignalScore(BaseModel):
    """Composite signal score for one ticker at one date."""
    date: date
    ticker: str
    composite_score: float = Field(ge=0, le=1)
    momentum_rank: float | None = None
    trend_rank: float | None = None
    lowvol_rank: float | None = None


# ── Optimization & Allocation ────────────────────────────────────────────────


class AllocationWeight(BaseModel):
    """Target weight for a single ticker."""
    ticker: str
    weight: float = Field(ge=0, le=1)
    method: str  # e.g. "hrp", "equal_weight"


class RebalanceTrade(BaseModel):
    """A single proposed rebalance trade."""
    ticker: str
    direction: str  # "BUY" | "SELL"
    current_weight: float
    target_weight: float
    delta_weight: float
    shares: float | None = None
    estimated_value: float | None = None


class RebalanceDecision(BaseModel):
    """Outcome of the rebalance decision logic."""
    should_rebalance: bool
    method: RebalanceMethod
    max_drift: float
    threshold: float
    reason: str


# ── Backtesting ──────────────────────────────────────────────────────────────


class BacktestResult(BaseModel):
    """Summary of a single backtest run."""
    strategy: str
    initial_capital: float
    final_value: float
    gross_cagr: float
    net_cagr: float
    sharpe: float
    max_drawdown: float
    total_costs: float
    total_taxes: float
    total_slippage: float
    turnover: float
    n_rebalances: int


class TradeRecord(BaseModel):
    """Single entry in the trade ledger."""
    date: date
    ticker: str
    direction: str
    shares: float
    price: float
    gross_value: float
    slippage: float
    brokerage: float
    stamp_duty: float
    tax: float
    net_value: float


# ── Monitoring & Alerts ──────────────────────────────────────────────────────


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftAlert(BaseModel):
    """Portfolio drift alert."""
    timestamp: datetime
    level: AlertLevel
    ticker: str
    current_weight: float
    target_weight: float
    drift: float
    message: str


class DrawdownAlert(BaseModel):
    """Drawdown breach alert."""
    timestamp: datetime
    level: AlertLevel
    current_drawdown: float
    threshold: float
    message: str


# ── API Response Models ──────────────────────────────────────────────────────


class PortfolioSummary(BaseModel):
    """API response for /portfolio/current."""
    as_of: date
    nav: float
    metrics: PortfolioMetrics
    holdings: list[AllocationWeight]
    regime: RegimeState | None = None


class RebalanceProposal(BaseModel):
    """API response for /rebalance/proposed."""
    decision: RebalanceDecision
    trades: list[RebalanceTrade]
    estimated_cost: float | None = None


class HealthStatus(BaseModel):
    """API response for /health."""
    status: str = "ok"
    version: str
    environment: str
    last_ingestion: datetime | None = None
    last_rebalance: datetime | None = None


# ── Execution Context ────────────────────────────────────────────────────────


class ExecutionContext(BaseModel):
    """Metadata for pipeline execution tracking."""
    execution_id: str
    pipeline: str
    started_at: datetime
    environment: str
    config_hash: str | None = None

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, v: str) -> str:
        if not v or len(v) < 8:
            raise ValueError("execution_id must be at least 8 characters")
        return v
