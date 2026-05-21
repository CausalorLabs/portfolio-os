"""
Monitoring — Audit & Traceability Layer.

Makes everything traceable:
  - Decision IDs for every rebalance, regime change, prediction, optimization
  - Event lineage: which inputs created which decisions
  - Trace graph construction
  - Compliance-ready audit export
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/monitoring.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Trace Events
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TraceEvent:
    """Single event in the lineage graph."""
    trace_id: str
    parent_id: str | None
    timestamp: datetime
    event_type: str       # ingestion | feature | prediction | optimization | execution | monitoring
    component: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return uuid.uuid4().hex[:12]


# ══════════════════════════════════════════════════════════════════════════════
# Audit Trail
# ══════════════════════════════════════════════════════════════════════════════


class AuditTrail:
    """
    Append-only audit trail with full event lineage.

    Every system decision gets a trace ID, and events are linked
    through parent-child relationships to form a lineage graph.
    """

    def __init__(self):
        cfg = _load_config().get("audit", {})
        self._events: list[TraceEvent] = []
        self._lineage_depth = cfg.get("lineage_depth", 5)
        self._active_traces: dict[str, str] = {}  # component → current trace_id

    def record_event(
        self,
        event_type: str,
        component: str,
        description: str,
        parent_id: str | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        duration_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> TraceEvent:
        """Record a system event with lineage information."""
        trace_id = generate_trace_id()

        # Auto-link to parent if available
        if parent_id is None:
            parent_id = self._active_traces.get(component)

        event = TraceEvent(
            trace_id=trace_id,
            parent_id=parent_id,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            component=component,
            description=description,
            inputs=inputs or [],
            outputs=outputs or [],
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        self._events.append(event)
        self._active_traces[component] = trace_id

        logger.debug(
            f"  Trace [{trace_id}] {event_type}/{component}: {description}"
        )
        return event

    def record_ingestion(
        self,
        source: str,
        n_records: int,
        duration_ms: float = 0.0,
        parent_id: str | None = None,
    ) -> TraceEvent:
        """Record a data ingestion event."""
        return self.record_event(
            "ingestion", "ingestion",
            f"Ingested {n_records} records from {source}",
            parent_id=parent_id,
            inputs=[source],
            outputs=["raw_data"],
            duration_ms=duration_ms,
            metadata={"source": source, "n_records": n_records},
        )

    def record_feature_computation(
        self,
        n_features: int,
        n_tickers: int,
        duration_ms: float = 0.0,
        parent_id: str | None = None,
    ) -> TraceEvent:
        """Record feature computation."""
        return self.record_event(
            "feature", "features",
            f"Computed {n_features} features for {n_tickers} tickers",
            parent_id=parent_id,
            inputs=["raw_data"],
            outputs=["feature_store"],
            duration_ms=duration_ms,
        )

    def record_prediction(
        self,
        model_name: str,
        n_predictions: int,
        mean_confidence: float,
        duration_ms: float = 0.0,
        parent_id: str | None = None,
    ) -> TraceEvent:
        """Record ML prediction."""
        return self.record_event(
            "prediction", "ml_inference",
            f"{model_name}: {n_predictions} predictions "
            f"(mean confidence: {mean_confidence:.2f})",
            parent_id=parent_id,
            inputs=["feature_store"],
            outputs=["alpha_scores"],
            duration_ms=duration_ms,
            metadata={
                "model": model_name,
                "n_predictions": n_predictions,
                "mean_confidence": mean_confidence,
            },
        )

    def record_optimization(
        self,
        method: str,
        n_assets: int,
        regime: str,
        duration_ms: float = 0.0,
        parent_id: str | None = None,
    ) -> TraceEvent:
        """Record portfolio optimization."""
        return self.record_event(
            "optimization", "optimization",
            f"{method} optimization for {n_assets} assets (regime: {regime})",
            parent_id=parent_id,
            inputs=["alpha_scores", "covariance_matrix", "regime_state"],
            outputs=["target_weights"],
            duration_ms=duration_ms,
            metadata={"method": method, "n_assets": n_assets, "regime": regime},
        )

    def record_execution_decision(
        self,
        decision: str,
        n_trades: int = 0,
        net_utility: float = 0.0,
        duration_ms: float = 0.0,
        parent_id: str | None = None,
    ) -> TraceEvent:
        """Record an execution decision."""
        return self.record_event(
            "execution", "execution",
            f"Decision: {decision} ({n_trades} trades, "
            f"net utility: {net_utility:.4f})",
            parent_id=parent_id,
            inputs=["target_weights", "current_portfolio"],
            outputs=["execution_journal"],
            duration_ms=duration_ms,
            metadata={
                "decision": decision,
                "n_trades": n_trades,
                "net_utility": net_utility,
            },
        )

    # ── Lineage Queries ──────────────────────────────────────────────────

    def get_lineage(self, trace_id: str) -> list[TraceEvent]:
        """
        Get full lineage chain for a trace ID.

        Walks up the parent chain to find all ancestor events.
        """
        chain = []
        current_id = trace_id
        depth = 0

        while current_id and depth < self._lineage_depth:
            event = next(
                (e for e in self._events if e.trace_id == current_id),
                None,
            )
            if event is None:
                break
            chain.append(event)
            current_id = event.parent_id
            depth += 1

        return chain

    def get_children(self, trace_id: str) -> list[TraceEvent]:
        """Get all direct children of a trace event."""
        return [e for e in self._events if e.parent_id == trace_id]

    def get_by_type(self, event_type: str) -> list[TraceEvent]:
        """Get all events of a specific type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_by_component(self, component: str) -> list[TraceEvent]:
        """Get all events from a specific component."""
        return [e for e in self._events if e.component == component]

    # ── Pipeline Trace ───────────────────────────────────────────────────

    def trace_pipeline_run(
        self,
        stages: list[dict],
    ) -> list[TraceEvent]:
        """
        Record a full pipeline run with linked stages.

        Each stage dict should have: event_type, component, description,
        and optional duration_ms, metadata.
        """
        events = []
        parent_id = None

        for stage in stages:
            event = self.record_event(
                event_type=stage.get("event_type", "pipeline"),
                component=stage.get("component", "unknown"),
                description=stage.get("description", ""),
                parent_id=parent_id,
                duration_ms=stage.get("duration_ms", 0),
                metadata=stage.get("metadata", {}),
            )
            events.append(event)
            parent_id = event.trace_id

        return events

    # ── Export ────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Export audit trail as DataFrame."""
        if not self._events:
            return pd.DataFrame()
        rows = [
            {
                "trace_id": e.trace_id,
                "parent_id": e.parent_id or "",
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "component": e.component,
                "description": e.description,
                "inputs": ",".join(e.inputs),
                "outputs": ",".join(e.outputs),
                "duration_ms": e.duration_ms,
            }
            for e in self._events
        ]
        return pd.DataFrame(rows)

    def recent(self, n: int = 50) -> list[TraceEvent]:
        """Get most recent trace events."""
        return sorted(self._events, key=lambda e: e.timestamp, reverse=True)[:n]

    def summary(self) -> dict:
        """Audit trail summary."""
        from collections import Counter
        types = Counter(e.event_type for e in self._events)
        components = Counter(e.component for e in self._events)
        return {
            "total_events": len(self._events),
            "by_type": dict(types),
            "by_component": dict(components),
            "active_traces": dict(self._active_traces),
        }

    def save(self, path: str = "data/exports/audit_trail.parquet") -> None:
        """Save audit trail to parquet."""
        df = self.to_dataframe()
        if not df.empty:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info(f"  Saved {len(df)} audit events to {path}")
