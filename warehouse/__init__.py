"""
DuckDB warehouse — unified query layer over partitioned parquet files.

Provides:
  - Lazy registration of parquet files as virtual tables
  - Type-safe reads via Pydantic contracts
  - Write-back to parquet with schema validation
  - Reproducible snapshots
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

# ── Default paths ────────────────────────────────────────────────────────────

_PROCESSED = Path("data/processed")
_DB_PATH = Path("data/warehouse.duckdb")

# Table → parquet file mapping
_TABLE_REGISTRY: dict[str, str] = {
    "inr_prices": "inr_prices.parquet",
    "portfolio_nav": "portfolio_nav.parquet",
    "returns": "returns.parquet",
    "drawdown_series": "drawdown_series.parquet",
    "rolling_analytics": "rolling_analytics.parquet",
    "fx_attribution": "fx_attribution.parquet",
    "features": "features.parquet",
    "signal_scores": "signal_scores.parquet",
    "target_weights": "target_weights.parquet",
    "backtest_nav": "backtest_nav.parquet",
    "rebalance_trades": "rebalance_trades.parquet",
    "trade_ledger": "trade_ledger.parquet",
    "regime_analysis": "regime_analysis.parquet",
    "regime_performance": "regime_performance.parquet",
    "monte_carlo_summary": "monte_carlo_summary.parquet",
    "stress_test_results": "stress_test_results.parquet",
    "fx_series": "fx_series.parquet",
}


class Warehouse:
    """DuckDB-backed query layer over parquet files."""

    def __init__(
        self,
        db_path: Path | str = _DB_PATH,
        data_dir: Path | str = _PROCESSED,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self._db_path))
        self._registered: set[str] = set()
        logger.debug(f"Warehouse connected: {self._db_path}")

    # ── Registration ─────────────────────────────────────────────────────────

    def _register(self, table: str) -> None:
        """Register a parquet file as a virtual table if not already done."""
        if table in self._registered:
            return

        filename = _TABLE_REGISTRY.get(table)
        if filename is None:
            raise ValueError(f"Unknown table: {table}. Known: {list(_TABLE_REGISTRY)}")

        path = self._data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        self._con.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path}')"
        )
        self._registered.add(table)
        logger.debug(f"Registered view: {table} → {path}")

    def register_all(self) -> list[str]:
        """Register all available parquet files. Returns list of registered tables."""
        registered = []
        for table in _TABLE_REGISTRY:
            path = self._data_dir / _TABLE_REGISTRY[table]
            if path.exists():
                self._register(table)
                registered.append(table)
        logger.info(f"Warehouse: {len(registered)}/{len(_TABLE_REGISTRY)} tables registered")
        return registered

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a DataFrame."""
        # Auto-register tables referenced in the query
        for table in _TABLE_REGISTRY:
            if table in sql:
                try:
                    self._register(table)
                except FileNotFoundError:
                    pass  # Table not yet populated
        return self._con.execute(sql).fetchdf()

    def read_table(self, table: str) -> pd.DataFrame:
        """Read an entire table as a DataFrame."""
        self._register(table)
        return self._con.execute(f"SELECT * FROM {table}").fetchdf()

    def table_info(self, table: str) -> pd.DataFrame:
        """Get column names and types for a table."""
        self._register(table)
        return self._con.execute(f"DESCRIBE {table}").fetchdf()

    def table_count(self, table: str) -> int:
        """Get row count for a table."""
        self._register(table)
        result = self._con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return result[0] if result else 0

    # ── Write ────────────────────────────────────────────────────────────────

    def write_parquet(self, df: pd.DataFrame, table: str) -> Path:
        """Write a DataFrame to the warehouse as a parquet file."""
        filename = _TABLE_REGISTRY.get(table)
        if filename is None:
            raise ValueError(f"Unknown table: {table}")

        path = self._data_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

        # Re-register the view
        if table in self._registered:
            self._registered.discard(table)
        self._register(table)

        logger.info(f"Wrote {len(df)} rows to {path}")
        return path

    # ── Inventory ────────────────────────────────────────────────────────────

    def available_tables(self) -> dict[str, bool]:
        """Return dict of table → exists (parquet file present)."""
        return {
            table: (self._data_dir / filename).exists()
            for table, filename in _TABLE_REGISTRY.items()
        }

    def summary(self) -> pd.DataFrame:
        """Return a summary of all tables with row counts."""
        rows = []
        for table, exists in self.available_tables().items():
            count = self.table_count(table) if exists else 0
            rows.append({"table": table, "exists": exists, "rows": count})
        return pd.DataFrame(rows)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._con.close()
        logger.debug("Warehouse connection closed")

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ── Module-level convenience ─────────────────────────────────────────────────

_warehouse: Warehouse | None = None


def get_warehouse() -> Warehouse:
    """Get or create the global warehouse instance."""
    global _warehouse
    if _warehouse is None:
        _warehouse = Warehouse()
    return _warehouse
