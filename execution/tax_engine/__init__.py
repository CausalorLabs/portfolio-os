"""
Execution Engine — Tax Engine.

Tracks:
  - FIFO lot-level realized gains
  - STCG vs LTCG classification (India/US rules)
  - Tax-loss harvesting opportunities
  - Tax-aware execution (avoid selling appreciated lots unless utility justifies it)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/execution_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class TaxLot:
    """A purchase lot for tax tracking."""
    ticker: str
    quantity: float
    cost_price: float
    purchase_date: date
    country: str = "IN"


@dataclass
class TaxImpact:
    """Tax impact of a proposed sale."""
    ticker: str
    quantity: float
    gain: float
    is_ltcg: bool
    holding_days: int
    tax_rate: float
    estimated_tax: float
    lot_details: list[dict] = field(default_factory=list)


# ── Tax Classification ──────────────────────────────────────────────────────


def classify_gain(
    purchase_date: date,
    sell_date: date,
    country: str = "IN",
) -> dict:
    """
    Classify a gain as STCG or LTCG.

    India: equity >12 months = LTCG (12.5%), else STCG (20%)
    US: >12 months = LTCG (15%), else STCG (15%)
    """
    cfg = _load_config().get("taxes", {})
    holding_days = (sell_date - purchase_date).days

    if country == "IN":
        india = cfg.get("india", {})
        period = india.get("holding_period_months", 12) * 30
        is_ltcg = holding_days >= period
        rate = india.get("ltcg_equity_pct", 12.5) if is_ltcg else india.get("stcg_equity_pct", 20.0)
        cess = india.get("cess_pct", 4.0)
        effective_rate = rate * (1 + cess / 100) / 100
    else:
        us = cfg.get("us", {})
        period = us.get("holding_period_months", 12) * 30
        is_ltcg = holding_days >= period
        rate = us.get("ltcg_pct", 15.0) if is_ltcg else us.get("stcg_pct", 15.0)
        effective_rate = rate / 100

    return {
        "holding_days": holding_days,
        "is_ltcg": is_ltcg,
        "tax_rate": effective_rate,
        "country": country,
    }


# ── FIFO Tax Impact ────────────────────────────────────────────────────────


def estimate_tax_on_sale(
    lots: list[TaxLot],
    sell_price: float,
    sell_qty: float,
    sell_date: date,
    country: str = "IN",
    cumulative_ltcg_used: float = 0.0,
) -> TaxImpact:
    """
    Estimate tax impact of selling using FIFO lot consumption.

    Applies India LTCG exemption (₹1.25L) if applicable.
    """
    cfg = _load_config().get("taxes", {})
    ltcg_exemption = cfg.get("india", {}).get("ltcg_exemption_inr", 125000)

    remaining = sell_qty
    total_tax = 0.0
    total_gain = 0.0
    lot_details = []
    ltcg_used = cumulative_ltcg_used

    for lot in lots:
        if remaining <= 0:
            break

        consumed = min(lot.quantity, remaining)
        gain = consumed * (sell_price - lot.cost_price)
        classification = classify_gain(lot.purchase_date, sell_date, country)

        taxable_gain = gain
        if classification["is_ltcg"] and country == "IN" and gain > 0:
            exempt = min(gain, max(0, ltcg_exemption - ltcg_used))
            taxable_gain = gain - exempt
            ltcg_used += exempt

        tax = max(0, taxable_gain * classification["tax_rate"])

        lot_details.append({
            "lot_date": lot.purchase_date,
            "lot_qty": consumed,
            "cost_price": lot.cost_price,
            "gain": gain,
            "is_ltcg": classification["is_ltcg"],
            "holding_days": classification["holding_days"],
            "tax": tax,
        })

        total_gain += gain
        total_tax += tax
        remaining -= consumed

    return TaxImpact(
        ticker=lots[0].ticker if lots else "",
        quantity=sell_qty,
        gain=total_gain,
        is_ltcg=all(d["is_ltcg"] for d in lot_details) if lot_details else False,
        holding_days=lot_details[0]["holding_days"] if lot_details else 0,
        tax_rate=total_tax / max(abs(total_gain), 1),
        estimated_tax=total_tax,
        lot_details=lot_details,
    )


# ── Tax-Loss Harvesting ────────────────────────────────────────────────────


def find_harvesting_opportunities(
    lots: dict[str, list[TaxLot]],
    current_prices: dict[str, float],
    sell_date: date,
    recently_sold: dict[str, date] | None = None,
) -> list[dict]:
    """
    Find tax-loss harvesting opportunities.

    Rules:
      - Lot must be at a loss
      - Loss must exceed minimum threshold
      - Must not violate wash sale rule (30-day window)
    """
    cfg = _load_config().get("taxes", {}).get("tax_loss_harvesting", {})

    if not cfg.get("enabled", True):
        return []

    min_loss = cfg.get("min_loss_to_harvest", 5000)
    wash_days = cfg.get("wash_sale_days", 30)
    max_annual = cfg.get("max_annual_harvest", 100000)

    if recently_sold is None:
        recently_sold = {}

    opportunities = []
    total_harvested = 0.0

    for ticker, ticker_lots in lots.items():
        price = current_prices.get(ticker)
        if price is None:
            continue

        # Wash sale check
        last_sold = recently_sold.get(ticker)
        if last_sold and (sell_date - last_sold).days < wash_days:
            continue

        for lot in ticker_lots:
            loss = lot.quantity * (price - lot.cost_price)
            if loss >= 0:
                continue

            abs_loss = abs(loss)
            if abs_loss < min_loss:
                continue

            if total_harvested + abs_loss > max_annual:
                continue

            classification = classify_gain(lot.purchase_date, sell_date, lot.country)

            opportunities.append({
                "ticker": ticker,
                "lot_date": lot.purchase_date,
                "quantity": lot.quantity,
                "cost_price": lot.cost_price,
                "current_price": price,
                "unrealized_loss": loss,
                "holding_days": classification["holding_days"],
                "is_ltcg": classification["is_ltcg"],
                "estimated_tax_savings": abs_loss * classification["tax_rate"],
            })

            total_harvested += abs_loss

    opportunities.sort(key=lambda x: x["unrealized_loss"])  # biggest losses first
    logger.info(f"  Tax-loss harvesting: {len(opportunities)} opportunities, "
                f"total loss={total_harvested:,.0f}")

    return opportunities


# ── Tax-Aware Sale Ordering ─────────────────────────────────────────────────


def rank_lots_for_sale(
    lots: list[TaxLot],
    sell_price: float,
    sell_date: date,
) -> list[dict]:
    """
    Rank lots by tax efficiency for selling.

    Prefer:
      1. Loss lots (harvest losses)
      2. LTCG lots (lower tax rate)
      3. Smallest gain lots
    """
    ranked = []
    for lot in lots:
        gain = lot.quantity * (sell_price - lot.cost_price)
        classification = classify_gain(lot.purchase_date, sell_date, lot.country)

        ranked.append({
            "lot": lot,
            "gain_per_unit": sell_price - lot.cost_price,
            "total_gain": gain,
            "is_ltcg": classification["is_ltcg"],
            "holding_days": classification["holding_days"],
            "tax_rate": classification["tax_rate"],
            "estimated_tax": max(0, gain * classification["tax_rate"]),
            # Sort key: losses first, then LTCG, then smallest gains
            "sort_key": (
                0 if gain < 0 else 1,           # losses first
                0 if classification["is_ltcg"] else 1,  # LTCG preferred
                gain,                             # smallest gains preferred
            ),
        })

    ranked.sort(key=lambda x: x["sort_key"])
    return ranked
