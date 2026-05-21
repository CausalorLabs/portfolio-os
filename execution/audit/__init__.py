"""
Execution Engine — Execution Journal / Audit Trail.

Logs EVERYTHING:
  - Why did we trade?
  - Why did we NOT trade?
  - What was expected utility?
  - What was the tax drag?

This becomes foundational for explainability, compliance,
and advisor workflows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
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
class AuditEntry:
    """Single execution journal entry."""
    timestamp: datetime
    decision_id: str
    action: str           # "trade" | "no_trade" | "harvest" | "evaluate"
    rationale: str
    expected_utility: float = 0.0
    cost_estimate: float = 0.0
    confidence: float = 0.0
    regime: str = "risk_on"
    trigger: str = ""     # "weight_drift" | "regime_change" | "risk_drift" etc.
    trades: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ExecutionJournal:
    """
    Append-only execution audit trail.

    Records every decision — trade and no-trade — with full context.
    """

    def __init__(self):
        self._entries: list[AuditEntry] = []
        cfg = _load_config().get("audit", {})
        self._log_all = cfg.get("log_all_decisions", True)
        self._include_rationale = cfg.get("include_rationale", True)

    def _gen_id(self) -> str:
        return str(uuid.uuid4())[:12]

    def log_trade(
        self,
        trades: list[dict],
        expected_utility: float = 0.0,
        cost_estimate: float = 0.0,
        confidence: float = 0.0,
        regime: str = "risk_on",
        trigger: str = "rebalance",
        rationale: str = "",
        metadata: dict | None = None,
    ) -> str:
        """Log a trade decision."""
        decision_id = self._gen_id()

        entry = AuditEntry(
            timestamp=datetime.now(),
            decision_id=decision_id,
            action="trade",
            rationale=rationale,
            expected_utility=expected_utility,
            cost_estimate=cost_estimate,
            confidence=confidence,
            regime=regime,
            trigger=trigger,
            trades=trades,
            metadata=metadata or {},
        )

        self._entries.append(entry)
        logger.info(f"  Journal: TRADE [{decision_id}] — {rationale[:80]}")
        return decision_id

    def log_no_trade(
        self,
        expected_utility: float = 0.0,
        cost_estimate: float = 0.0,
        confidence: float = 0.0,
        regime: str = "risk_on",
        trigger: str = "none",
        rationale: str = "",
        metadata: dict | None = None,
    ) -> str:
        """Log a no-trade decision (equally important)."""
        if not self._log_all:
            return ""

        decision_id = self._gen_id()

        entry = AuditEntry(
            timestamp=datetime.now(),
            decision_id=decision_id,
            action="no_trade",
            rationale=rationale,
            expected_utility=expected_utility,
            cost_estimate=cost_estimate,
            confidence=confidence,
            regime=regime,
            trigger=trigger,
            metadata=metadata or {},
        )

        self._entries.append(entry)
        logger.debug(f"  Journal: NO_TRADE [{decision_id}] — {rationale[:80]}")
        return decision_id

    def log_harvest(
        self,
        opportunities: list[dict],
        rationale: str = "",
    ) -> str:
        """Log a tax-loss harvesting decision."""
        decision_id = self._gen_id()

        entry = AuditEntry(
            timestamp=datetime.now(),
            decision_id=decision_id,
            action="harvest",
            rationale=rationale,
            metadata={"opportunities": opportunities},
        )

        self._entries.append(entry)
        logger.info(f"  Journal: HARVEST [{decision_id}] — {len(opportunities)} lots")
        return decision_id

    # ── Queries ─────────────────────────────────────────────────────────

    def recent(self, n: int = 10) -> list[AuditEntry]:
        """Last N journal entries."""
        return self._entries[-n:]

    def trades_only(self) -> list[AuditEntry]:
        """Only trade decisions."""
        return [e for e in self._entries if e.action == "trade"]

    def no_trades_only(self) -> list[AuditEntry]:
        """Only no-trade decisions."""
        return [e for e in self._entries if e.action == "no_trade"]

    def by_regime(self, regime: str) -> list[AuditEntry]:
        """Filter by regime."""
        return [e for e in self._entries if e.regime == regime]

    def summary(self) -> dict:
        """Journal summary stats."""
        n_total = len(self._entries)
        n_trade = sum(1 for e in self._entries if e.action == "trade")
        n_no = sum(1 for e in self._entries if e.action == "no_trade")
        n_harvest = sum(1 for e in self._entries if e.action == "harvest")

        avg_utility = 0.0
        if n_trade > 0:
            avg_utility = sum(
                e.expected_utility for e in self._entries if e.action == "trade"
            ) / n_trade

        return {
            "total_entries": n_total,
            "trades": n_trade,
            "no_trades": n_no,
            "harvests": n_harvest,
            "trade_skip_ratio": n_no / max(n_total, 1),
            "avg_trade_utility": avg_utility,
        }

    # ── Export ──────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Export journal as DataFrame."""
        if not self._entries:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": e.timestamp,
                "decision_id": e.decision_id,
                "action": e.action,
                "rationale": e.rationale if self._include_rationale else "",
                "expected_utility": e.expected_utility,
                "cost_estimate": e.cost_estimate,
                "confidence": e.confidence,
                "regime": e.regime,
                "trigger": e.trigger,
                "n_trades": len(e.trades),
            }
            for e in self._entries
        ])

    def save(self, path: str = "data/processed/execution_journal.parquet"):
        """Save journal to parquet."""
        df = self.to_dataframe()
        if not df.empty:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info(f"Execution journal saved: {path} ({len(df)} entries)")
