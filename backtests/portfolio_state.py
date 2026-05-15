"""
Portfolio state engine — maintains the true evolving portfolio state
through the backtest: holdings, cash, cost basis, realized/unrealized PnL.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


@dataclass
class TaxLot:
    """A single purchase lot for cost-basis tracking."""

    ticker: str
    quantity: float
    cost_price: float  # per-unit price in INR
    purchase_date: pd.Timestamp


@dataclass
class PortfolioState:
    """
    Full mutable portfolio state.

    Tracks actual holdings (with tax lots), cash balance,
    cumulative costs/taxes, and realized PnL.
    """

    cash: float = 0.0
    holdings: dict[str, float] = field(default_factory=dict)  # ticker → qty
    tax_lots: dict[str, list[TaxLot]] = field(default_factory=dict)
    realized_pnl: float = 0.0
    total_costs_paid: float = 0.0
    total_taxes_paid: float = 0.0

    # ── queries ──────────────────────────────────────────────────────────

    def nav(self, prices: dict[str, float]) -> float:
        """Portfolio NAV = cash + market value of all holdings."""
        market_value = sum(
            qty * prices.get(t, 0.0) for t, qty in self.holdings.items()
        )
        return self.cash + market_value

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Current portfolio weights (including cash)."""
        total = self.nav(prices)
        if total <= 0:
            return {}
        w = {t: qty * prices.get(t, 0.0) / total for t, qty in self.holdings.items()}
        w["_CASH"] = self.cash / total
        return w

    def unrealized_pnl(self, prices: dict[str, float]) -> dict[str, float]:
        """Unrealized PnL per ticker."""
        pnl = {}
        for ticker, lots in self.tax_lots.items():
            current = prices.get(ticker, 0.0)
            pnl[ticker] = sum(
                lot.quantity * (current - lot.cost_price) for lot in lots
            )
        return pnl

    def cost_basis(self, ticker: str) -> float:
        """Weighted average cost basis for a ticker."""
        lots = self.tax_lots.get(ticker, [])
        total_qty = sum(lot.quantity for lot in lots)
        if total_qty <= 0:
            return 0.0
        return sum(lot.quantity * lot.cost_price for lot in lots) / total_qty

    # ── mutations ────────────────────────────────────────────────────────

    def buy(
        self,
        ticker: str,
        quantity: float,
        price: float,
        date: pd.Timestamp,
        cost: float = 0.0,
    ) -> None:
        """Execute a buy: deduct cash, add holdings and tax lot."""
        total_cost = quantity * price + cost
        self.cash -= total_cost
        self.total_costs_paid += cost

        self.holdings[ticker] = self.holdings.get(ticker, 0.0) + quantity

        if ticker not in self.tax_lots:
            self.tax_lots[ticker] = []
        self.tax_lots[ticker].append(
            TaxLot(ticker=ticker, quantity=quantity, cost_price=price, purchase_date=date)
        )

    def sell(
        self,
        ticker: str,
        quantity: float,
        price: float,
        date: pd.Timestamp,
        cost: float = 0.0,
        tax: float = 0.0,
    ) -> float:
        """
        Execute a sell: add cash (net of costs/tax), reduce holdings,
        consume tax lots FIFO. Returns realized PnL.
        """
        proceeds = quantity * price - cost - tax
        self.cash += proceeds
        self.total_costs_paid += cost
        self.total_taxes_paid += tax

        self.holdings[ticker] = self.holdings.get(ticker, 0.0) - quantity
        if self.holdings[ticker] <= 1e-9:
            self.holdings.pop(ticker, None)

        # Consume lots FIFO
        realized = self._consume_lots(ticker, quantity, price)
        self.realized_pnl += realized

        return realized

    def _consume_lots(self, ticker: str, sell_qty: float, sell_price: float) -> float:
        """FIFO lot consumption. Returns realized PnL."""
        lots = self.tax_lots.get(ticker, [])
        remaining = sell_qty
        realized = 0.0
        new_lots = []

        for lot in lots:
            if remaining <= 0:
                new_lots.append(lot)
                continue

            if lot.quantity <= remaining:
                realized += lot.quantity * (sell_price - lot.cost_price)
                remaining -= lot.quantity
            else:
                realized += remaining * (sell_price - lot.cost_price)
                new_lots.append(
                    TaxLot(
                        ticker=ticker,
                        quantity=lot.quantity - remaining,
                        cost_price=lot.cost_price,
                        purchase_date=lot.purchase_date,
                    )
                )
                remaining = 0

        self.tax_lots[ticker] = new_lots
        if not new_lots:
            self.tax_lots.pop(ticker, None)

        return realized

    def snapshot(self, date: pd.Timestamp, prices: dict[str, float]) -> dict:
        """Capture current state as a flat dict for recording."""
        total_nav = self.nav(prices)
        unrealized = self.unrealized_pnl(prices)
        return {
            "date": date,
            "cash": self.cash,
            "market_value": total_nav - self.cash,
            "nav": total_nav,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": sum(unrealized.values()),
            "total_costs": self.total_costs_paid,
            "total_taxes": self.total_taxes_paid,
            "n_positions": len(self.holdings),
        }

    def clone(self) -> PortfolioState:
        """Deep copy for branching simulations."""
        return copy.deepcopy(self)
