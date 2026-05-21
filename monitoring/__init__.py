"""
Structured logging, observability & monitoring orchestrator.

Provides:
  - Execution ID tracking across pipeline runs
  - Structured log format with JSON output option
  - Pipeline timing context manager
  - Step-level instrumentation
  - MonitoringEngine: unified orchestrator for Sprint 6 subsystems
"""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from loguru import logger

# ── Execution context ────────────────────────────────────────────────────────

_current_execution_id: str | None = None
_current_pipeline: str | None = None


def new_execution_id() -> str:
    """Generate a new execution ID."""
    return uuid.uuid4().hex[:12]


def get_execution_id() -> str:
    """Get the current execution ID, or create one."""
    global _current_execution_id
    if _current_execution_id is None:
        _current_execution_id = new_execution_id()
    return _current_execution_id


def set_execution_context(execution_id: str, pipeline: str) -> None:
    """Set the current execution context."""
    global _current_execution_id, _current_pipeline
    _current_execution_id = execution_id
    _current_pipeline = pipeline


# ── Logger setup ─────────────────────────────────────────────────────────────


def _format_record(record: dict) -> str:
    """Custom log format with execution ID."""
    exec_id = _current_execution_id or "------"
    pipeline = _current_pipeline or "system"
    elapsed = record.get("elapsed", "")
    return (
        "<green>{time:HH:mm:ss}</green> "
        f"<cyan>[{exec_id}]</cyan> "
        f"<blue>{pipeline}</blue> | "
        "<level>{level: <8}</level> | "
        "{message}\n"
    )


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure loguru with structured format."""
    logger.remove()

    if json_output:
        logger.add(
            sys.stderr,
            level=level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=_format_record,
            colorize=True,
        )

    # File output (always JSON for machine parsing)
    logger.add(
        "logs/pipeline_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        serialize=True,
        rotation="1 day",
        retention="30 days",
        compression="gz",
    )


# ── Pipeline timing ─────────────────────────────────────────────────────────


@contextmanager
def pipeline_context(name: str) -> Generator[str, None, None]:
    """Context manager that sets execution context and times the pipeline."""
    exec_id = new_execution_id()
    set_execution_context(exec_id, name)

    logger.info(f"Pipeline started: {name}")
    logger.info(f"Execution ID: {exec_id}")
    logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    start = time.perf_counter()
    try:
        yield exec_id
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error(f"Pipeline FAILED after {elapsed:.1f}s: {name}")
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(f"Pipeline completed in {elapsed:.1f}s: {name}")
    finally:
        global _current_execution_id, _current_pipeline
        _current_execution_id = None
        _current_pipeline = None


@contextmanager
def step_timer(step_name: str) -> Generator[None, None, None]:
    """Time an individual pipeline step."""
    logger.info(f"▸ {step_name}")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error(f"  ✗ {step_name} FAILED ({elapsed:.1f}s)")
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(f"  ✓ {step_name} ({elapsed:.1f}s)")


# ── Monitoring Engine (Sprint 6) ─────────────────────────────────────────────


class MonitoringEngine:
    """
    Unified orchestrator for all Sprint 6 monitoring subsystems.

    Ties together:
      - Attribution (performance + factor)
      - Explainability (decisions + narratives)
      - Alerts (portfolio, risk, regime, ML, operational)
      - Notifications (Telegram, Slack, email)
      - Observability (component + model health)
      - Anomaly detection (portfolio, model, execution)
      - Audit trail (event lineage + traceability)
    """

    def __init__(self):
        from monitoring.alerts import AlertEngine
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        from monitoring.audit import AuditTrail
        from monitoring.explainability import DecisionTimeline
        from monitoring.notifications import NotificationDispatcher
        from monitoring.observability import ObservabilityEngine

        self.alerts = AlertEngine()
        self.anomalies = AnomalyDetectionEngine()
        self.audit = AuditTrail()
        self.timeline = DecisionTimeline()
        self.notifications = NotificationDispatcher()
        self.observability = ObservabilityEngine()

        logger.info("MonitoringEngine initialized")

    def run_health_check(self) -> dict:
        """Run all health checks and return combined status."""
        stale = self.observability.check_staleness()
        health = self.observability.summary()
        alert_summary = self.alerts.summary()
        anomaly_summary = self.anomalies.summary()

        return {
            "system_health": health,
            "alerts": alert_summary,
            "anomalies": anomaly_summary,
            "stale_components": stale,
        }

    def run_monitoring_cycle(
        self,
        weights: dict[str, float] | None = None,
        target_weights: dict[str, float] | None = None,
        current_vol: float = 0.0,
        target_vol: float = 0.15,
        current_drawdown: float = 0.0,
        current_regime: str = "risk_on",
        regime_changed: bool = False,
        previous_regime: str | None = None,
        ml_confidence: float = 0.5,
        turnover: float = 0.0,
        nav_return: float | None = None,
    ) -> dict:
        """
        Run a full monitoring cycle: alerts + anomalies + health check.

        Dispatches notifications for any alerts fired.
        """
        # 1. Alert checks
        alerts = self.alerts.run_all_checks(
            weights=weights,
            target_weights=target_weights,
            current_vol=current_vol,
            target_vol=target_vol,
            current_drawdown=current_drawdown,
            current_regime=current_regime,
            regime_changed=regime_changed,
            previous_regime=previous_regime,
            ml_confidence=ml_confidence,
        )

        # 2. Anomaly checks
        anomalies = self.anomalies.check_portfolio_anomalies(
            turnover=turnover,
            nav_return=nav_return,
        )

        # 3. Dispatch notifications for alerts
        for alert in alerts:
            self.notifications.dispatch_alert(alert)

        # 4. Record in audit trail
        self.audit.record_event(
            "monitoring", "monitoring",
            f"Cycle complete: {len(alerts)} alerts, {len(anomalies)} anomalies",
        )

        # 5. Record success
        self.observability.record_success("monitoring", message="Cycle complete")

        return {
            "alerts_fired": len(alerts),
            "anomalies_detected": len(anomalies),
            "alerts": [a.title for a in alerts],
            "anomalies": [a.description for a in anomalies],
        }

    def summary(self) -> dict:
        """Combined monitoring summary."""
        return {
            "alerts": self.alerts.summary(),
            "anomalies": self.anomalies.summary(),
            "observability": self.observability.summary(),
            "notifications": self.notifications.summary(),
            "audit": self.audit.summary(),
            "timeline": self.timeline.summary(),
        }
