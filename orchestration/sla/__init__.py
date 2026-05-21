"""
Orchestration — SLA (Service Level Agreement) Layer.

Pipeline SLA monitoring:
  - Per-stage timing vs targets
  - Uptime tracking
  - SLA compliance reporting
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/orchestration.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class SLACheck:
    """Result of an SLA check."""
    component: str
    target_minutes: float
    actual_minutes: float
    met: bool
    timestamp: datetime


class SLATracker:
    """
    SLA compliance tracker for pipeline operations.
    """

    def __init__(self):
        cfg = _load_config().get("sla", {})
        self._targets = {
            "ingestion": cfg.get("ingestion_max_minutes", 10),
            "features": cfg.get("features_max_minutes", 5),
            "ml_inference": cfg.get("ml_inference_max_minutes", 3),
            "risk": cfg.get("risk_max_minutes", 5),
            "total_pipeline": cfg.get("total_pipeline_max_minutes", 30),
        }
        self._staleness_hours = cfg.get("staleness_hours", 24)
        self._target_uptime = cfg.get("target_uptime_pct", 99.0)
        self._checks: list[SLACheck] = []
        self._runs: list[dict] = []  # {timestamp, success, duration_min}

    def record_stage(
        self,
        component: str,
        duration_seconds: float,
        target_override: float | None = None,
    ) -> SLACheck:
        """Record a stage execution and check SLA compliance."""
        actual_min = duration_seconds / 60.0
        target_min = target_override or self._targets.get(component, 30)

        check = SLACheck(
            component=component,
            target_minutes=target_min,
            actual_minutes=actual_min,
            met=actual_min <= target_min,
            timestamp=datetime.now(timezone.utc),
        )
        self._checks.append(check)

        if not check.met:
            logger.warning(
                f"  SLA breach: {component} took {actual_min:.1f}min (target: {target_min}min)"
            )
        return check

    def record_run(self, success: bool, duration_minutes: float) -> None:
        """Record a full pipeline run."""
        self._runs.append({
            "timestamp": datetime.now(timezone.utc),
            "success": success,
            "duration_min": duration_minutes,
        })

    def check_staleness(self, last_update: datetime | None = None) -> bool:
        """Check if data is stale (beyond staleness threshold)."""
        if last_update is None:
            return True
        elapsed = datetime.now(timezone.utc) - last_update
        return elapsed > timedelta(hours=self._staleness_hours)

    def get_uptime_pct(self, days: int = 30) -> float:
        """Calculate uptime percentage over last N days."""
        if not self._runs:
            return 100.0
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [r for r in self._runs if r["timestamp"] >= cutoff]
        if not recent:
            return 100.0
        successes = sum(1 for r in recent if r["success"])
        return (successes / len(recent)) * 100

    def get_compliance_report(self) -> pd.DataFrame:
        """Get SLA compliance report."""
        if not self._checks:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "component": c.component,
                "target_min": c.target_minutes,
                "actual_min": c.actual_minutes,
                "met": c.met,
                "timestamp": c.timestamp,
            }
            for c in self._checks
        ])

    def summary(self) -> dict:
        total = len(self._checks)
        met = sum(1 for c in self._checks if c.met)
        return {
            "total_checks": total,
            "sla_met": met,
            "sla_breaches": total - met,
            "compliance_pct": (met / total * 100) if total else 100.0,
            "uptime_pct": self.get_uptime_pct(),
        }
