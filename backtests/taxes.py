"""
Tax engine — models realistic capital gains taxes with lot tracking.

Indian tax rules (POC approximation):
  Equity STCG (<1 year) : 15%
  Equity LTCG (>1 year) : 10%

US approximation:
  Flat cap-gains rate   : 15%

Taxes are hidden negative alpha — reducing turnover can outperform
many predictive models.
"""

import pandas as pd
from loguru import logger


# ── Tax rate schedules ───────────────────────────────────────────────────────

TAX_RATES = {
    "IN": {
        "stcg_rate": 0.15,           # short-term < 12 months
        "ltcg_rate": 0.10,           # long-term > 12 months
        "ltcg_exemption": 100_000,   # ₹1L annual LTCG exemption
        "holding_threshold_days": 365,
    },
    "US": {
        "stcg_rate": 0.15,
        "ltcg_rate": 0.15,
        "ltcg_exemption": 0,
        "holding_threshold_days": 365,
    },
}


def calculate_capital_gains_tax(
    sell_price: float,
    cost_price: float,
    quantity: float,
    purchase_date: pd.Timestamp,
    sell_date: pd.Timestamp,
    country: str,
    cumulative_ltcg_exemption_used: float = 0.0,
) -> dict:
    """
    Calculate capital gains tax for a single lot sale.

    Parameters
    ----------
    sell_price : float
        Per-unit sale price in INR.
    cost_price : float
        Per-unit cost basis in INR.
    quantity : float
    purchase_date, sell_date : pd.Timestamp
    country : str
        "IN" or "US".
    cumulative_ltcg_exemption_used : float
        How much of the annual LTCG exemption is already used.

    Returns
    -------
    dict
        Keys: gain, holding_days, is_ltcg, tax_rate, taxable_gain, tax
    """
    rates = TAX_RATES.get(country, TAX_RATES["US"])

    gain = quantity * (sell_price - cost_price)
    holding_days = (sell_date - purchase_date).days
    is_ltcg = holding_days >= rates["holding_threshold_days"]

    if gain <= 0:
        # No tax on losses (can be used for harvesting later)
        return {
            "gain": gain,
            "holding_days": holding_days,
            "is_ltcg": is_ltcg,
            "tax_rate": 0.0,
            "taxable_gain": 0.0,
            "tax": 0.0,
        }

    if is_ltcg:
        tax_rate = rates["ltcg_rate"]
        exemption = rates["ltcg_exemption"]
        remaining_exemption = max(0, exemption - cumulative_ltcg_exemption_used)
        taxable_gain = max(0, gain - remaining_exemption)
    else:
        tax_rate = rates["stcg_rate"]
        taxable_gain = gain

    tax = taxable_gain * tax_rate

    return {
        "gain": gain,
        "holding_days": holding_days,
        "is_ltcg": is_ltcg,
        "tax_rate": tax_rate,
        "taxable_gain": taxable_gain,
        "tax": tax,
    }


def calculate_tax_on_lots(
    lots: list,
    sell_price: float,
    sell_qty: float,
    sell_date: pd.Timestamp,
    country: str,
) -> tuple[float, list[dict]]:
    """
    Calculate total tax for a FIFO sell across multiple lots.

    Parameters
    ----------
    lots : list[TaxLot]
        Tax lots (from PortfolioState).
    sell_price : float
        Per-unit sell price in INR.
    sell_qty : float
    sell_date : pd.Timestamp
    country : str

    Returns
    -------
    tuple[float, list[dict]]
        (total_tax, list of per-lot tax details)
    """
    remaining = sell_qty
    total_tax = 0.0
    details = []
    cumulative_ltcg_used = 0.0

    for lot in lots:
        if remaining <= 0:
            break

        qty_from_lot = min(lot.quantity, remaining)
        result = calculate_capital_gains_tax(
            sell_price=sell_price,
            cost_price=lot.cost_price,
            quantity=qty_from_lot,
            purchase_date=lot.purchase_date,
            sell_date=sell_date,
            country=country,
            cumulative_ltcg_exemption_used=cumulative_ltcg_used,
        )

        if result["is_ltcg"] and result["gain"] > 0:
            cumulative_ltcg_used += result["gain"]

        total_tax += result["tax"]
        details.append({
            "lot_date": lot.purchase_date,
            "lot_qty": qty_from_lot,
            "cost_price": lot.cost_price,
            **result,
        })
        remaining -= qty_from_lot

    return total_tax, details


def summarize_tax_impact(tax_details: list[dict]) -> dict:
    """Aggregate tax impact across all trades in a period."""
    total_tax = sum(d.get("tax", 0.0) for d in tax_details)
    total_stcg_tax = sum(d["tax"] for d in tax_details if not d.get("is_ltcg", False))
    total_ltcg_tax = sum(d["tax"] for d in tax_details if d.get("is_ltcg", False))
    total_gains = sum(d.get("gain", 0.0) for d in tax_details)
    total_losses = sum(d["gain"] for d in tax_details if d.get("gain", 0.0) < 0)

    return {
        "total_tax": total_tax,
        "stcg_tax": total_stcg_tax,
        "ltcg_tax": total_ltcg_tax,
        "total_gains": total_gains,
        "total_losses": total_losses,
        "effective_tax_rate": total_tax / total_gains if total_gains > 0 else 0.0,
    }
