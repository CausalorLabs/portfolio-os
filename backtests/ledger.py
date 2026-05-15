"""
Trade ledger — full audit trail of every trade executed in the backtest.

Critical for debugging, explainability, and future compliance.
"""

from pathlib import Path

import pandas as pd
from loguru import logger


class TradeLedger:
    """Append-only trade log."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(
        self,
        date: pd.Timestamp,
        ticker: str,
        action: str,
        quantity: float,
        market_price: float,
        execution_price: float,
        slippage_cost: float,
        transaction_cost: float,
        tax: float,
        realized_pnl: float,
        country: str,
    ) -> None:
        """Record a single executed trade."""
        self._records.append({
            "date": date,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "market_price": market_price,
            "execution_price": execution_price,
            "notional": execution_price * quantity,
            "slippage_cost": slippage_cost,
            "transaction_cost": transaction_cost,
            "tax": tax,
            "realized_pnl": realized_pnl,
            "country": country,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Return ledger as DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=[
                "date", "ticker", "action", "quantity", "market_price",
                "execution_price", "notional", "slippage_cost",
                "transaction_cost", "tax", "realized_pnl", "country",
            ])
        return pd.DataFrame(self._records)

    def save(self, path: str | Path = "data/processed/trade_ledger.parquet") -> None:
        """Persist to parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info(f"Trade ledger saved → {path}  ({len(df)} trades)")

    def summary(self) -> dict:
        """Aggregate ledger statistics."""
        df = self.to_dataframe()
        if df.empty:
            return {"n_trades": 0}
        return {
            "n_trades": len(df),
            "n_buys": (df["action"] == "BUY").sum(),
            "n_sells": (df["action"] == "SELL").sum(),
            "total_notional": df["notional"].sum(),
            "total_slippage": df["slippage_cost"].sum(),
            "total_costs": df["transaction_cost"].sum(),
            "total_taxes": df["tax"].sum(),
            "total_friction": df["slippage_cost"].sum() + df["transaction_cost"].sum() + df["tax"].sum(),
            "total_realized_pnl": df["realized_pnl"].sum(),
        }
